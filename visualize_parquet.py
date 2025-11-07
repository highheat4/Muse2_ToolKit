import os
import sys
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import datetime as _dt

def _parse_time_spec(value: Optional[str], default_seconds: Optional[float] = None) -> Optional[float]:
    """
    Parse a time specification into seconds.
    Accepts plain numbers (interpreted as seconds) or with suffix: s, m, h (case-insensitive).
    Returns None if value is None and default_seconds is None.
    """
    if value is None:
        return default_seconds
    s = str(value).strip().lower()
    if not s:
        return default_seconds
    try:
        # raw seconds (int/float string)
        return float(s)
    except Exception:
        pass
    mult = 1.0
    if s.endswith("s"):
        s = s[:-1]
        mult = 1.0
    elif s.endswith("m"):
        s = s[:-1]
        mult = 60.0
    elif s.endswith("h"):
        s = s[:-1]
        mult = 3600.0
    try:
        return float(s) * mult
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid time spec: {value}. Use seconds or add suffix s/m/h (e.g., 240, 4m).")


def _choose_columns(df: pd.DataFrame, explicit: Optional[List[str]], prefer_clean: bool) -> List[str]:
    if explicit:
        present = [c for c in explicit if c in df.columns]
        if not present:
            raise ValueError("None of the requested columns exist in the Parquet file.")
        return present
    # Prefer cleaned EEG if available, else raw EEG; else all numeric (excluding timestamp)
    eeg_clean = [c for c in df.columns if c.startswith("eeg_clean_")]
    eeg_raw = [c for c in df.columns if c.startswith("eeg_raw_")]
    if prefer_clean and eeg_clean:
        return eeg_clean
    if eeg_raw:
        return eeg_raw
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != "timestamp"]


def _compute_window(df: pd.DataFrame, start_spec: Optional[str], end_spec: Optional[str], duration_spec: Optional[str]) -> Tuple[float, float]:
    if "timestamp" not in df.columns:
        raise ValueError("Parquet must contain a 'timestamp' column (seconds).")
    ts = df["timestamp"].astype(float)
    if ts.empty:
        raise ValueError("Parquet has no rows.")
    min_ts = float(ts.min())
    max_ts = float(ts.max())
    end_seconds = _parse_time_spec(end_spec, default_seconds=max_ts)
    dur_seconds = _parse_time_spec(duration_spec, default_seconds=240.0)
    start_seconds = _parse_time_spec(start_spec, default_seconds=end_seconds - dur_seconds)
    # Clamp to available range
    if start_seconds is None or end_seconds is None:
        raise ValueError("Unable to compute time window.")
    if start_seconds < min_ts:
        start_seconds = min_ts
    if end_seconds > max_ts:
        end_seconds = max_ts
    if end_seconds <= start_seconds:
        raise ValueError("End time must be greater than start time.")
    return float(start_seconds), float(end_seconds)


def visualize(parquet_path: str,
              start: Optional[str],
              end: Optional[str],
              duration: Optional[str],
              columns: Optional[List[str]],
              prefer_clean: bool,
              absolute_time: bool,
              output_path: Optional[str],
              width: float,
              height: float,
              dpi: int,
              title: Optional[str],
              show_legend: bool,
              show_clean: bool,
              show_raw: bool,
              show_ppg: bool,
              show_gyro: bool,
              show_acc: bool,
              interactive: bool) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.dates as mdates  # type: ignore
        from matplotlib.widgets import RangeSlider, CheckButtons  # type: ignore
    except Exception:
        print("matplotlib is required. Install it with: pip install matplotlib")
        sys.exit(1)

    parquet_path = os.path.abspath(parquet_path)
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Failed to read Parquet: {e}")
        sys.exit(1)

    try:
        start_s, end_s = _compute_window(df, start, end, duration)
    except Exception as e:
        print(f"Invalid time window: {e}")
        sys.exit(1)

    window_df = df[(df["timestamp"] >= start_s) & (df["timestamp"] <= end_s)].copy()
    if window_df.empty:
        print("No data in the requested time window.")
        sys.exit(1)

    # Determine groups
    groups = {
        "eeg_clean": [c for c in window_df.columns if c.startswith("eeg_clean_")] if show_clean else [],
        "eeg_raw":   [c for c in window_df.columns if c.startswith("eeg_raw_")]   if show_raw   else [],
        "ppg":       [c for c in window_df.columns if c.startswith("ppg_")]       if show_ppg   else [],
        "gyro":      [c for c in window_df.columns if c.startswith("gyro_")]      if show_gyro  else [],
        "acc":       [c for c in window_df.columns if c.startswith("acc_")]       if show_acc   else [],
    }

    # If explicit columns requested, override groups to only include those
    if columns:
        present = [c for c in columns if c in window_df.columns]
        if not present:
            print("None of the requested columns exist in the Parquet file.")
            sys.exit(1)
        # Place requested columns into their matching group; ignore unknowns
        groups = {k: [c for c in present if c.startswith(f"{k}_")] for k in groups.keys()}

    # Reorder EEG channels to canonical Muse order if possible
    def _order_eeg(cols: List[str]) -> List[str]:
        order = ["TP9", "AF7", "AF8", "TP10"]
        pref: List[str] = []
        remaining = list(cols)
        for name in order:
            for c in cols:
                if c.endswith("_" + name):
                    pref.append(c)
                    if c in remaining:
                        remaining.remove(c)
        # Add any others deterministically
        pref.extend(sorted(remaining))
        return pref

    groups["eeg_clean"] = _order_eeg(groups["eeg_clean"])
    groups["eeg_raw"] = _order_eeg(groups["eeg_raw"])

    # Compute total rows and build figure
    plot_order = ["eeg_clean", "eeg_raw", "ppg", "gyro", "acc"]  # gyro/acc at bottom
    rows = sum(len(groups[g]) for g in plot_order)
    if rows == 0:
        print("No recognized columns to plot.")
        sys.exit(1)

    # X axis handling
    x_sec = window_df["timestamp"].to_numpy(dtype=float)
    if absolute_time:
        x = pd.to_datetime(x_sec, unit="s")
        x_is_datetime = True
    else:
        x = x_sec - float(start_s)
        x_is_datetime = False

    # Dynamic height so traces are readable
    min_height_per_row = 1.0
    computed_h = max(height, rows * min_height_per_row)
    fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(width, computed_h), dpi=dpi)
    if rows == 1:
        axes = [axes]  # normalize

    # Plot each channel in its own axis
    idx = 0
    lines = {}
    axes_by_col = {}
    channels_order: List[str] = []
    for group_name in plot_order:
        cols = groups[group_name]
        for col in cols:
            ax = axes[idx]
            y = window_df[col].to_numpy(dtype=float)
            line, = ax.plot(x, y, linewidth=0.9, color="#2c7fb8")  # single color per channel axis
            lines[col] = line
            axes_by_col[col] = ax
            # Keep channel order list for relayout in plotting order
            channels_order.append(col)
            # Channel label with RMS as in muselsl-style annotation
            rms = float(np.sqrt(np.mean(np.square(y)))) if y.size else 0.0
            chan = col.split("_", 1)[1] if "_" in col else col
            ax.set_ylabel(f"{chan} - {rms:.2f}", rotation=0, ha="right", va="center")
            ax.yaxis.set_label_coords(-0.02, 0.5)
            ax.grid(True, linestyle="-", alpha=0.15)
            ax.set_yticks([])  # muselsl hides y ticks
            # Thinner borders
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            idx += 1

    # Titles and x-axis formatting
    if title:
        fig.suptitle(title, y=0.995)
    else:
        base = os.path.basename(parquet_path)
        span = end_s - start_s
        fig.suptitle(f"{base} — window {span:.1f}s", y=0.995)

    if x_is_datetime:
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axes[-1].set_xlabel("time (HH:MM:SS)")
    else:
        axes[-1].set_xlabel("time (s)")

    if show_legend:
        # Legend isn't typical for muselsl view since each axis is one channel; skip by default
        for ax in axes:
            ax.legend().set_visible(False)
    # Leave room at bottom for slider if interactive
    bottom_pad = 0.30 if interactive else 0.06
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(bottom=bottom_pad)

    # Relayout helper: resize and re-stack visible axes to fill space
    def _relayout_axes() -> None:
        left, right = 0.08, 0.98
        top = 0.98
        bottom = bottom_pad
        # Determine visible columns in original order
        visible_cols = [c for c in channels_order if lines[c].get_visible()]
        n = len(visible_cols)
        # Hide all axes first
        for c in channels_order:
            axes_by_col[c].set_visible(False)
        if n == 0:
            fig.canvas.draw_idle()
            return
        gap = 0.01
        avail_h = top - bottom - gap * (n - 1)
        height = max(0.02, avail_h / n)
        width = right - left
        # Layout from top to bottom in channel order
        for i, c in enumerate(visible_cols):
            # position index from top
            pos_from_top = i
            # convert to bottom-based coordinates
            y = bottom + (n - 1 - pos_from_top) * (height + gap)
            ax = axes_by_col[c]
            ax.set_visible(True)
            ax.set_position([left, y, width, height])
            # show x labels only on bottom-most visible axis
            show_xticks = (i == n - 1)
            ax.tick_params(labelbottom=show_xticks)
            if absolute_time and show_xticks:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.canvas.draw_idle()

    # Initial relayout to ensure even spacing
    _relayout_axes()

    # Interactive time window slider
    if interactive:
        min_ts = float(df["timestamp"].min())
        max_ts = float(df["timestamp"].max())
        slider_ax = fig.add_axes([0.12, 0.05, 0.76, 0.03])
        slider = RangeSlider(slider_ax, "Time", min_ts, max_ts, valinit=(start_s, end_s))
        # Keep strong references to widgets to prevent garbage collection
        if not hasattr(fig, "_muse_widgets"):
            fig._muse_widgets = {}  # type: ignore[attr-defined]
        fig._muse_widgets["slider"] = slider  # type: ignore[index]

        def _update_window(val):
            s, e = slider.val
            # slice once
            seg = df[(df["timestamp"] >= s) & (df["timestamp"] <= e)]
            if seg.empty:
                return
            # update each line
            x_seg = seg["timestamp"].to_numpy(dtype=float)
            if absolute_time:
                x_draw = pd.to_datetime(x_seg, unit="s")
                xlim = pd.to_datetime([s, e], unit="s")
            else:
                x_draw = x_seg - float(s)
                xlim = (0.0, float(e - s))
            for col, line in lines.items():
                if col not in seg.columns:
                    # Column might be missing in the new window (unlikely); skip update
                    continue
                y_seg = seg[col].to_numpy(dtype=float)
                line.set_data(x_draw, y_seg)
                ax = axes_by_col[col]
                ax.set_xlim(xlim)
                # update RMS label
                rms_val = float(np.sqrt(np.mean(np.square(y_seg)))) if y_seg.size else 0.0
                chan_name = col.split("_", 1)[1] if "_" in col else col
                ax.set_ylabel(f"{chan_name} - {rms_val:.2f}", rotation=0, ha="right", va="center")
                ax.relim()
                ax.autoscale_view(scalex=False, scaley=True)
            # update title span
            if title is None:
                fig.suptitle(f"{os.path.basename(parquet_path)} — window {float(e - s):.1f}s", y=0.995)
            fig.canvas.draw_idle()

        slider.on_changed(_update_window)

        # Checkboxes to toggle columns per group
        non_empty_groups = [g for g in plot_order if len(groups[g]) > 0]
        if non_empty_groups:
            left = 0.08
            right = 0.92
            total_w = right - left
            seg_w = total_w / float(len(non_empty_groups))
            cb_y = 0.10
            cb_h = 0.14
            label_to_col = {}

            def _mk_label(col_name: str) -> str:
                return col_name.split("_", 1)[1] if "_" in col_name else col_name

            for i, gname in enumerate(non_empty_groups):
                gcols = groups[gname]
                if not gcols:
                    continue
                ax_cb = fig.add_axes([left + i * seg_w + 0.01, cb_y, seg_w - 0.02, cb_h])
                labels = [_mk_label(c) for c in gcols]
                actives = [lines[c].get_visible() for c in gcols]
                cb = CheckButtons(ax_cb, labels, actives=actives)
                ax_cb.set_title(gname.replace("_", " ").upper(), fontsize=9, loc="left")
                # Map labels to fully-qualified column names within this group
                for lab, col in zip(labels, gcols):
                    label_to_col[(gname, lab)] = col

                def _on_clicked_factory(group_key: str):
                    def _on_clicked(label: str):
                        key = (group_key, label)
                        if key not in label_to_col:
                            return
                        col_name = label_to_col[key]
                        line = lines[col_name]
                        vis = not line.get_visible()
                        line.set_visible(vis)
                        # Relayout all axes to fill space and redraw
                        _relayout_axes()
                        fig.canvas.draw_idle()
                    return _on_clicked

                cb.on_clicked(_on_clicked_factory(gname))
                # Keep reference so callbacks remain active
                if "checkboxes" not in fig._muse_widgets:  # type: ignore[attr-defined]
                    fig._muse_widgets["checkboxes"] = []  # type: ignore[index]
                fig._muse_widgets["checkboxes"].append(cb)  # type: ignore[index]

    if output_path:
        out = os.path.abspath(output_path)
        try:
            fig.savefig(out, bbox_inches="tight")
            print(f"Wrote figure: {out}")
        except Exception as e:
            print(f"Failed to save figure: {e}")
            sys.exit(1)
    else:
        plt.show()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize a time segment from an aggregated Muse2 Parquet file.")
    p.add_argument("parquet", help="Path to the aggregated Parquet file.")
    p.add_argument("--start", help="Window start (seconds or with suffix s/m/h). Default: end - duration.")
    p.add_argument("--end", help="Window end (seconds or with suffix s/m/h). Default: last timestamp.")
    p.add_argument("--duration", default="240s", help="Window length (seconds or s/m/h). Default: 240s (last 4 minutes).")
    p.add_argument("--columns", help="Comma-separated list of columns to plot. Overrides group selection.")
    p.add_argument("--prefer-raw", action="store_true", help="Prefer eeg_raw_* if both cleaned and raw exist (only used with --columns absent).")
    p.add_argument("--absolute-time", action="store_true", help="Use absolute timestamps on x-axis (recommended for muselsl-style).")
    p.add_argument("--output", help="Path to save the plot image (e.g., out.png). If omitted, shows an interactive window.")
    p.add_argument("--width", type=float, default=12.0, help="Figure width in inches (default 12).")
    p.add_argument("--height", type=float, default=6.0, help="Figure height in inches (default 6).")
    p.add_argument("--dpi", type=int, default=120, help="Figure DPI (default 120).")
    p.add_argument("--title", help="Custom plot title.")
    p.add_argument("--no-legend", action="store_true", help="Hide legend.")
    p.add_argument("--no-clean", action="store_true", help="Hide cleaned EEG subplots.")
    p.add_argument("--no-raw", action="store_true", help="Hide raw EEG subplots.")
    p.add_argument("--no-ppg", action="store_true", help="Hide PPG subplots.")
    p.add_argument("--no-gyro", action="store_true", help="Hide gyro subplots.")
    p.add_argument("--no-acc", action="store_true", help="Hide accelerometer subplots.")
    p.add_argument("--static", action="store_true", help="Disable interactive time slider.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    visualize(
        parquet_path=args.parquet,
        start=args.start,
        end=args.end,
        duration=args.duration,
        columns=cols,
        prefer_clean=not args.prefer_raw,
        absolute_time=args.absolute_time or True,  # default to absolute for muselsl-style
        output_path=args.output,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title=args.title,
        show_legend=not args.no_legend,
        show_clean=not args.no_clean,
        show_raw=not args.no_raw,
        show_ppg=not args.no_ppg,
        show_gyro=not args.no_gyro,
        show_acc=not args.no_acc,
        interactive=not args.static,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


