from muselsl import stream, list_muses, record
from threading import Thread
import time
import os
import asyncio

muses = list_muses()

def stream_muse():
    """Thread function to stream Muse data (sets its own asyncio loop for Bleak)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        stream(muses[0]['address'], ppg_enabled=True, acc_enabled=True, gyro_enabled=True)
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
    time.sleep(10)

    # Start recording threads for EEG, PPG, ACC, and GYRO
    modalities = ["EEG", "PPG", "ACC", "GYRO"]
    record_threads = [
        Thread(target=record_modality, args=(source, session_dir, RECORD_DURATION))
        for source in modalities
    ]

    for t in record_threads:
        t.start()

    for t in record_threads:
        t.join()