#!/bin/bash

# Activate environment 
conda env create -f environment.yml
conda activate radar

# Start capturing data
python -m src/streaming.start_radar_1

# Lauch real-time visualization 
python -m src/streaming.stream_1.py