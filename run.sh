#!/bin/bash

sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

SRC_PATH=./src

single_scripts=("preprocess" "train" "inference" "super_resolution" "create_video" "split_video" "remove_background" "overlap_results")

clear

python3.10 "$SRC_PATH/preprocess.py"
python3.10 "$SRC_PATH/train.py"
python3.10 "$SRC_PATH/inference.py"
python3.10 "$SRC_PATH/super_resolution.py"
python3.10 "$SRC_PATH/create_video.py"