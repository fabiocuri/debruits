#!/bin/bash

get_value() {
    local field=$1
    local value=$(awk -v field="$field" '$1 == field {print $2}' "./config.yaml")
    echo "$value"
}

PIP_VERSION=$(get_value "PIP_VERSION:")
PYTHON_VERSION=$(get_value "PYTHON_VERSION:")
SRC_PATH=$(get_value "SRC_PATH:")
SCRIPT_FOLDER=$(get_value "SCRIPT_FOLDER:")

virtualenv env
source env/bin/activate
$PIP_VERSION install -r requirements.txt

single_scripts=("preprocess" "train" "inference" "overlap_results" "split_video" "remove_background")
folder_scripts=("super_resolution" "create_video")
all_scripts=("all")

if [[ " ${single_scripts[@]} " =~ " ${1} " ]]; then
    $PYTHON_VERSION "$SRC_PATH/$1.py"
elif [[ " ${folder_scripts[@]} " =~ " ${1} " ]]; then
    $PYTHON_VERSION "$SRC_PATH/$1.py" $SCRIPT_FOLDER
elif [[ " ${all_scripts[@]} " =~ " ${1} " ]]; then
    $PYTHON_VERSION "$SRC_PATH/preprocess.py"
    $PYTHON_VERSION "$SRC_PATH/train.py"
    $PYTHON_VERSION "$SRC_PATH/inference.py"
    $PYTHON_VERSION "$SRC_PATH/super_resolution.py" plot
    $PYTHON_VERSION "$SRC_PATH/super_resolution.py" inference
    $PYTHON_VERSION "$SRC_PATH/create_video.py" plot
    $PYTHON_VERSION "$SRC_PATH/create_video.py" inference
else
    echo "Error: Invalid argument provided."
fi