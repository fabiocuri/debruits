#!/bin/sh
N_EPOCHS=1
N_SUPER_RESOLUTION=2
BRIGHTNESS=10
CONTRAST=1
BLUR=10
SATURATION=3
PYTHON_VERSION=python3
PIP_VERSION=pip3
SUPER_RESOLUTION_FOLDER="/content/drive/MyDrive/plots2/cropped"
TO_BE_CROPPED_FOLDER="/content/drive/MyDrive/plots2"

if [ "$1" = "create" ]; then
 $PIP_VERSION install -r /content/debruits/requirements.txt
elif [[ "$1" = "preprocess" ]]; then
 $PYTHON_VERSION /content/debruits/src/preprocess.py train $BRIGHTNESS $CONTRAST $BLUR $SATURATION
 $PYTHON_VERSION /content/debruits/src/preprocess.py val $BRIGHTNESS $CONTRAST $BLUR $SATURATION
elif [[ "$1" = "train" ]]; then
 $PYTHON_VERSION /content/debruits/src/train.py $1 $N_EPOCHS
elif [[ "$1" = "inference" ]]; then
 $PYTHON_VERSION /content/debruits/src/preprocess.py test $BRIGHTNESS $CONTRAST $BLUR $SATURATION
 $PYTHON_VERSION /content/debruits/src/inference.py
elif [[ "$1" = "super_resolution" ]]; then
 $PYTHON_VERSION /content/debruits/src/super_resolution.py $N_SUPER_RESOLUTION $SUPER_RESOLUTION_FOLDER
elif [[ "$1" = "crop" ]]; then
 $PYTHON_VERSION /content/debruits/src/crop.py $TO_BE_CROPPED_FOLDER
fi
