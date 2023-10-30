#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Error: Please provide one of the following arguments: preprocess, train, super-resolution-plots, create-video-plots, inference, super-resolution-inference, create-video-inference..."
    exit 1
fi

# Get the argument value
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

# Install requirements
virtualenv env
source env/bin/activate
$PIP_VERSION install -r requirements.txt

# Execute based on the provided argument
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
    *)
        echo "Error: Invalid argument provided."
        exit 1
        ;;
esac
