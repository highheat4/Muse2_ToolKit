import os
import sys
import argparse
import numpy as np
import pandas as pd
import shutil
from typing import Optional, Dict, List


def _spread_duplicate_timestamps(df: pd.DataFrame, expected_group_size: int = 3) -> pd.DataFrame:
    """
    For streams like ACC/GYRO that repeat the same coarse timestamp multiple times
    (e.g., three samples with identical tenths-second timestamps), spread the
    duplicates evenly before the next unique timestamp so that timestamps become
    unique and ordered:

    - Keep the first sample's timestamp unchanged
    - Distribute the remaining (n-1) samples evenly in (t0, t1), where t1 is the
      next unique base timestamp; if no t1 exists, use a fallback gap
    """
    if 'timestamps' not in df.columns:
        return df

    ts = df['timestamps'].astype(float).to_numpy()
    if ts.size == 0:
        return df

    # Estimate a reasonable fallback gap from unique base timestamps
    base = pd.Series(ts).drop_duplicates().to_numpy()
    diffs = np.diff(base)
    diffs = diffs[diffs > 0]
    fallback_gap = float(np.median(diffs)) if diffs.size else 0.1

    new_ts = ts.copy()

    i = 0
    n_total = ts.size
    while i < n_total:
        t0 = ts[i]
        j = i + 1
        # Find run of identical timestamps
        while j < n_total and ts[j] == t0:
            j += 1
        n_run = j - i

        # Determine the next base timestamp
        if j < n_total:
            t1 = ts[j]
        else:
            t1 = t0 + fallback_gap

        gap = t1 - t0
        if gap <= 0:
            gap = fallback_gap

        # Spread evenly in (t0, t1), keeping the first unchanged
        if n_run > 1:
            step = gap / float(max(n_run, 1))
            for k in range(1, n_run):
                new_ts[i + k] = t0 + step * k

        i = j

    df = df.copy()
    df['timestamps'] = new_ts
    return df


def _load_and_bucket_csv(csv_path: str, prefix: str, decimals: int = 3) -> Optional[pd.DataFrame]:
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if 'timestamps' not in df.columns:
        return None

    df['timestamps'] = df['timestamps'].astype(float)
    # For ACC and GYRO, spread duplicate coarse timestamps before bucketing
    if prefix.lower() in ('acc', 'gyro'):
        df = _spread_duplicate_timestamps(df, expected_group_size=3)
    # Quantize to desired decimal places (default 3 = milliseconds)
    df['bucket'] = np.round(df['timestamps'], decimals)

    # Select only numeric columns for aggregation; exclude timestamps and bucket
    numeric_cols = [c for c in df.columns if c not in ['timestamps', 'bucket']]
    # Sanitize column names (remove spaces) and add prefix
    rename_map: Dict[str, str] = {}
    for c in numeric_cols:
        clean = c.replace(' ', '')
        rename_map[c] = f"{prefix}_{clean}"
    df = df.rename(columns=rename_map)

    # Drop timestamps explicitly so it cannot overlap during joins later
    if 'timestamps' in df.columns:
        df = df.drop(columns=['timestamps'])

    # Group by bucket, take mean of all numeric columns (excluding timestamps)
    grouped = df.groupby('bucket').mean(numeric_only=True)
    # Ensure the index is named 'bucket'
    grouped.index.name = 'bucket'
    return grouped


def _align_to_target_index(df: pd.DataFrame, target_index: pd.Index) -> pd.DataFrame:
    """Reindex to target_index and linearly interpolate across the time index; then ffill/bfill edges."""
    if df is None or df.empty:
        return pd.DataFrame(index=target_index)
    aligned = df.reindex(target_index)
    aligned = aligned.interpolate(method='index')
    aligned = aligned.ffill().bfill()
    return aligned


def aggregate_session(session_dir: str, output_root: Optional[str] = None, cleanup: bool = True) -> Optional[str]:
    """
    Aggregate EEG/PPG/ACC/GYRO CSVs in a session directory into a single CSV.
    The output CSV is written as data/<session_timestamp>.csv by default.
    Returns the output CSV path, or None if nothing was aggregated.
    """
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        print(f"Session directory not found: {session_dir}")
        return None

    session_name = os.path.basename(session_dir.rstrip(os.sep))
    data_root = output_root or os.path.dirname(session_dir)
    os.makedirs(data_root, exist_ok=True)
    output_path = os.path.join(data_root, f"{session_name}.csv")

    eeg_path = os.path.join(session_dir, 'eeg.csv')
    ppg_path = os.path.join(session_dir, 'ppg.csv')
    acc_path = os.path.join(session_dir, 'acc.csv')
    gyro_path = os.path.join(session_dir, 'gyro.csv')

    eeg = _load_and_bucket_csv(eeg_path, 'eeg', decimals=3)
    if eeg is None or eeg.empty:
        print("EEG is required to define the 3-decimal target timeline; none found.")
        return None

    target_index = eeg.index

    ppg = _load_and_bucket_csv(ppg_path, 'ppg', decimals=3)
    acc = _load_and_bucket_csv(acc_path, 'acc', decimals=3)
    gyro = _load_and_bucket_csv(gyro_path, 'gyro', decimals=3)

    combined = eeg.copy()
    for part in [ppg, acc, gyro]:
        aligned = _align_to_target_index(part, target_index)
        combined = combined.join(aligned)

    combined = combined.sort_index()
    combined = combined.reset_index().rename(columns={'bucket': 'timestamp'})

    # Quantize all numeric outputs (including timestamp) to 3 decimals
    try:
        combined['timestamp'] = combined['timestamp'].astype(float).round(3)
    except Exception:
        pass
    combined = combined.round(3)

    combined.to_csv(output_path, index=False, float_format='%.3f')
    print(f"Wrote aggregated CSV: {output_path}")
    
    # Clean up the session directory after successful aggregation
    if cleanup:
        try:
            shutil.rmtree(session_dir)
            print(f"Cleaned up session directory: {session_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up session directory {session_dir}: {e}")
    
    return output_path


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Aggregate a Muse recording session directory into a single CSV by 0.01s buckets.')
    parser.add_argument('session', help='Path to session directory (e.g., data/1762365268) or just the timestamp (e.g., 1762365268).')
    parser.add_argument('--data-root', default=None, help='Root data directory containing the session folder. If not provided and session is a timestamp, defaults to ./data')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    session_arg = args.session
    if os.path.isdir(session_arg):
        session_dir = session_arg
        data_root = args.data_root or os.path.dirname(os.path.abspath(session_dir))
    else:
        data_root = args.data_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        session_dir = os.path.join(data_root, session_arg)

    result = aggregate_session(session_dir, output_root=data_root)
    return 0 if result else 1


if __name__ == '__main__':
    sys.exit(main())


