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

single_scripts=("preprocess" "train" "inference" "super_resolution" "create_video" "overlap_results" "split_video" "remove_background")

clear

python "$SRC_PATH/$1.py"