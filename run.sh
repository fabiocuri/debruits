#!/bin/bash

get_value() {
    local field=$1
    local value=$(awk -v field="$field" '$1 == field {print $2}' "./config.yaml")
    echo "$value"
}

PIP_VERSION=$(get_value "PIP_VERSION:")
PYTHON_VERSION=$(get_value "PYTHON_VERSION:")
SRC_PATH=$(get_value "SRC_PATH:")

virtualenv env
source env/bin/activate
$PIP_VERSION install -r requirements.txt

case "$1" in
    "preprocess")
        $PYTHON_VERSION "$SRC_PATH/preprocess.py"
        ;;
    "train")
        $PYTHON_VERSION "$SRC_PATH/train.py"
        ;;
    "super-resolution-plot")
        $PYTHON_VERSION "$SRC_PATH/super_resolution.py" plot
        ;;
    "create-video-plot")
        $PYTHON_VERSION "$SRC_PATH/create_video.py" plot
        ;;
    "inference")
        $PYTHON_VERSION "$SRC_PATH/inference.py"
        ;;
    "super-resolution-inference")
        $PYTHON_VERSION "$SRC_PATH/super_resolution.py" inference
        ;;
    "create-video-inference")
        $PYTHON_VERSION "$SRC_PATH/frames2video.py" inference
        ;;
    "overlap-results")
        $PYTHON_VERSION "$SRC_PATH/overlap_results.py"
        ;;
    "split-video")
        $PYTHON_VERSION "$SRC_PATH/split_video.py" /home/fabio/Videos/test.mp4
        ;;
    "remove-background")
        $PYTHON_VERSION "$SRC_PATH/remove_background.py" /home/fabio/Videos/test
        ;;
    "all")
        $PYTHON_VERSION "$SRC_PATH/preprocess.py"
        $PYTHON_VERSION "$SRC_PATH/train.py"
        $PYTHON_VERSION "$SRC_PATH/super_resolution.py" plots
        $PYTHON_VERSION "$SRC_PATH/frames2video.py" plots
        $PYTHON_VERSION "$SRC_PATH/inference.py"
        $PYTHON_VERSION "$SRC_PATH/super_resolution.py" inference
        $PYTHON_VERSION "$SRC_PATH/frames2video.py" inference
        ;;
    *)
        echo "Error: Invalid argument provided."
        exit 1
        ;;
esac
