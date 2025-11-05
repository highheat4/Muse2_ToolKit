from muselsl import stream, list_muses, record
from threading import Thread, Event
import time
import os
import asyncio
import atexit
import signal
from merge_session_data import aggregate_session
import subprocess
import sys

muses = list_muses()
shutdown_event = Event()


def play_tone():
    """Play a simple notification tone on macOS; fallback to terminal bell."""
    try:
        subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'], check=False)
    except Exception:
        try:
            # Terminal bell as last resort
            print('\a', end='', flush=True)
        except Exception:
            pass

def stream_muse():
    """Thread function to stream Muse data (sets its own asyncio loop for Bleak)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        try:
            stream(muses[0]['address'], ppg_enabled=True, acc_enabled=True, gyro_enabled=True)
        except Exception:
            if not shutdown_event.is_set():
                play_tone()
            return
        else:
            # stream() returned (likely disconnected)
            if not shutdown_event.is_set():
                play_tone()
    finally:
        try:
            loop.close()
        except Exception:
            pass

RECORD_DURATION = 10800


def record_modality(data_source, output_dir, duration=RECORD_DURATION):
    """Record a specific modality (EEG/PPG/ACC/GYRO) to CSV into output_dir for given duration."""
    filename = os.path.join(output_dir, f"{data_source.lower()}.csv")
    record(duration=duration, filename=filename, data_source=data_source)

if __name__ == "__main__":
    if not muses:
        raise RuntimeError("No Muse devices found. Ensure the headset is on and discoverable.")

    # Create per-session directory under data/<unix_timestamp>
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_dir, 'data')
    session_ts = str(int(time.time()))
    session_dir = os.path.join(data_root, session_ts)
    os.makedirs(session_dir, exist_ok=True)

    # Start streaming first so the EEG LSL outlet exists before recording resolves it
    stream_thread = Thread(target=stream_muse, daemon=True)
    stream_thread.start()

    # Give the stream a moment to initialize its LSL outlets
    time.sleep(15)

    # Start recording threads for EEG, PPG, ACC, and GYRO
    modalities = ["EEG", "PPG", "ACC", "GYRO"]
    record_threads = [
        Thread(target=record_modality, args=(source, session_dir, RECORD_DURATION))
        for source in modalities
    ]

    # Ensure aggregation runs on exit or interruption
    _aggregated = {"done": False}

    def _write_aggregate_once():
        if not _aggregated["done"]:
            try:
                aggregate_session(session_dir, output_root=data_root, cleanup=True)
            finally:
                _aggregated["done"] = True

    def _handle_exit(signum=None, frame=None):
        shutdown_event.set()
        _write_aggregate_once()
        raise SystemExit(0)

    atexit.register(_write_aggregate_once)
    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)
    try:
        signal.signal(signal.SIGHUP, _handle_exit)
        signal.signal(signal.SIGQUIT, _handle_exit)
    except Exception:
        pass

    for t in record_threads:
        t.start()

    # Watch the streaming thread; if it ends (disconnect), aggregate without cleanup
    def _watch_stream_and_aggregate():
        stream_thread.join()
        if not shutdown_event.is_set():
            try:
                aggregate_session(session_dir, output_root=data_root, cleanup=False)
            except Exception as e:
                print(f"Warning: Early aggregation failed: {e}")

    watcher_thread = Thread(target=_watch_stream_and_aggregate, daemon=True)
    watcher_thread.start()

    # Listen for 'exit'/'quit' on stdin to force aggregation and terminate
    def _watch_stdin_for_exit():
        try:
            for line in sys.stdin:
                if line.strip().lower() in ('exit', 'quit', 'q'):
                    shutdown_event.set()
                    _write_aggregate_once()
                    os._exit(0)
        except Exception:
            pass

    stdin_thread = Thread(target=_watch_stdin_for_exit, daemon=True)
    stdin_thread.start()

    for t in record_threads:
        t.join()

    # After recordings finish, aggregate the session
    shutdown_event.set()
    _write_aggregate_once()