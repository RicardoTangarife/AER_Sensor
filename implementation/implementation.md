# Acoustic Event Recognition (AER) Sensor Implementation

This section details the implementation of an acoustic event recognition sensor that uses machine learning models to detect specific acoustic events such as gunshots, sirens, and screams. The sensor captures audio, processes it, and triggers appropriate responses based on detected events. The implementation is designed to manage storage efficiently and respond to acoustic events with high confidence levels.

## Requirements

### Mounting the SD Card

The SD card needs to be mounted to a specific directory to be passed as a parameter to the script. Here is how to mount the SD card in Linux:

```sh
sudo mkdir -p /mnt/sdcard
sudo mount /dev/mmcblk1p1 /mnt/sdcard

df -h
```
Mount on always on restart, add line to file:
```sh
sudo nano /etc/fstab

/dev/mmcblk1p1 /mnt/sdcard vfat defaults 0 2
```
Replace /dev/mmcblk1p1 with the correct identifier for your device.


## Implementation Overview

The implementation consists of several Python scripts organized into different modules:

### AER_ARQ4_TL_ind.py
This script contains the implementation of the sensor's main functionality. It records audio, processes it using machine learning models, and triggers appropriate actions based on the detected acoustic events.

### audio_processing.py
This module provides functions for preparing and extracting features from audio files. It resamples the audio to a specified sample rate, amplifies it, and extracts Mel-frequency cepstral coefficients (MFCCs) using the librosa library.

### storage_manager.py
This module manages storage space on the device. It calculates the free space available, monitors the size of the segments and events folders, and deletes old files if the storage capacity is exceeded. Additionally, it provides functions to save audio files with timestamp-based filenames.

### AER_ARQ1_TL_fused_pll.py
An alternative implementation that executes tasks in parallel for improved efficiency. It uses fused models for two of the acoustic events.

### AER_ARQ1_TL_fused.py
Similar to the previous script but with a sequential execution of tasks.


## How to Use the Scripts

### AER_ARQ4_TL_ind.py

### Dependencies
- TensorFlow
- Librosa
- NumPy
- Sounddevice
- SciPy

### Usage
```sh
sudo python3 AER_ARQ4_TL_ind.py [--sd_path SD_PATH] [--conf1 CONF1] [--conf2 CONF2] [--conf3 CONF3] [--max_size_segments MAX_SIZE_SEGMENTS] [--max_size_events MAX_SIZE_EVENTS] [--days_to_keep DAYS_TO_KEEP]
```

### Arguments

- `--sd_path`: Path to the mounted SD card (default: "/mnt/sdcard").
- `--conf1`: Confidence level for event 1 detection (default: 0.97).
- `--conf2`: Confidence level for event 2 detection (default: 0.97).
- `--conf3`: Confidence level for event 3 detection (default: 0.80).
- `--max_size_segments`: Maximum size in GB for the segments folder (default: 2.0).
- `--max_size_events`: Maximum size in GB for the events folder (default: 1.0).
- `--days_to_keep`: Number of days to keep stored data after reaching maximum capacity (default: 0).


### Example Usage

```sh
sudo python3 AER_ARQ4_TL_ind.py --sd_path /mnt/sdcard --conf1 0.97 --conf2 0.97 --conf3 0.80 --max_size_segments 2.0 --max_size_events 1.0 --days_to_keep 0
```

This command runs the AER_ARQ4_TL_ind.py script with default parameters, recording audio from mic connected with USB and save the record in the specified SD card path and using predefined confidence levels for event detection. Detected events trigger appropriate actions and are saved in the segments and events folders, with old files being deleted if the storage capacity is exceeded.