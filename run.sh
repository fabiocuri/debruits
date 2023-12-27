#!/bin/bash

sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

get_value() {
    local field=$1
    local value=$(awk -v field="$field" '$1 == field {print $2}' "./config.yaml")
    echo "$value"
}

SRC_PATH=$(get_value "SRC_PATH:")
SCRIPT_FOLDER=$(get_value "SCRIPT_FOLDER:")

single_scripts=("preprocess" "train" "inference" "overlap_results" "split_video" "remove_background")
folder_scripts=("super_resolution" "create_video")
all_scripts=("all")

clear

if [[ " ${single_scripts[@]} " =~ " ${1} " ]]; then
    python "$SRC_PATH/$1.py"
elif [[ " ${folder_scripts[@]} " =~ " ${1} " ]]; then
    python "$SRC_PATH/$1.py" $SCRIPT_FOLDER
elif [[ " ${all_scripts[@]} " =~ " ${1} " ]]; then
    python "$SRC_PATH/preprocess.py"
    python "$SRC_PATH/train.py"
    python "$SRC_PATH/inference.py"
    python "$SRC_PATH/super_resolution.py" plot
    python "$SRC_PATH/super_resolution.py" inference
    python "$SRC_PATH/create_video.py" plot
    python "$SRC_PATH/create_video.py" inference
else
    echo "Error: Invalid argument provided."
fi