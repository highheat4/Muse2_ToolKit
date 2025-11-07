# Muse2_ToolKit
Tools to capture, clean, aggregate, and visualize data from a Muse 2 headset, with a focus on enabling personal analysis of focus and meditative states. museLSL alone doesn't offer the end-to-end workflow I want, and BrainFlow's Muse support often requires extra hardware. This toolkit bridges that gap by:
- Storing raw multi-stream data (EEG, PPG, ACC, GYRO) and cleaned EEG in a single, compact Parquet file per session
- Providing a reconnect-resilient recorder and offline EEG cleaning (notch + bandpass) using BrainFlow
- Offering a muselsl-style visualization CLI that lets you explore any time window, toggle channels, and interactively resize the view
- Laying the groundwork for future analytics on focus/meditation metrics


# Commits

## V0.2
Used brainflow to add cleaned eeg columns. Data stored in parquet files now for better compression. Added basic visualization (python visualize_parquet.py /path/to/parquet).

## V0.1
Noticed granularity differences in streamed data; EEG data is the most granular, so interpolated data from the other three files and combined csv readings into one output. Linear interpolation for ACC, GYRO, and PPG data. 

Also added handling for Muse disconnect to store the data easily.

## V0
Currently, just stores the data streamed from my Muse 2 to the data folder.

