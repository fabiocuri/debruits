#!/bin/bash

argument="$1"

src_path="./src"
yaml_file="./config.yaml"

get_value() {
    local field=$1
    local value=$(awk -v field="$field" '$1 == field {print $2}' "$yaml_file")
    echo "$value"
}

PIP_VERSION=$(get_value "PIP_VERSION:")
PYTHON_VERSION=$(get_value "PYTHON_VERSION:")

virtualenv env
source env/bin/activate
$PIP_VERSION install -r requirements.txt

case "$argument" in
    "preprocess")
        $PYTHON_VERSION "$src_path/preprocess.py"
        ;;
    "train")
        $PYTHON_VERSION "$src_path/train.py"
        ;;
    "super-resolution-plots")
        $PYTHON_VERSION "$src_path/super_resolution.py" plots
        ;;
    "create-video-plots")
        $PYTHON_VERSION "$src_path/frames2video.py" plots
        ;;
    "inference")
        $PYTHON_VERSION "$src_path/inference.py"
        ;;
    "super-resolution-inference")
        $PYTHON_VERSION "$src_path/super_resolution.py" inference
        ;;
    "create-video-inference")
        $PYTHON_VERSION "$src_path/frames2video.py" inference
        ;;
    "overlap-results")
        $PYTHON_VERSION "$src_path/overlap_results.py"
        ;;
    "split-video")
        $PYTHON_VERSION "$src_path/split_video.py" /home/fabio/Videos/untitled.mpg
        ;;
    "all")
        $PYTHON_VERSION "$src_path/preprocess.py"
        $PYTHON_VERSION "$src_path/train.py"
        $PYTHON_VERSION "$src_path/super_resolution.py" plots
        $PYTHON_VERSION "$src_path/frames2video.py" plots
        $PYTHON_VERSION "$src_path/inference.py"
        $PYTHON_VERSION "$src_path/super_resolution.py" inference
        $PYTHON_VERSION "$src_path/frames2video.py" inference
        ;;
    *)
        echo "Error: Invalid argument provided."
        exit 1
        ;;
esac
