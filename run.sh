#!/bin/sh
N_EPOCHS=50
N_SUPER_RESOLUTION=2
BRIGHTNESS=2
CONTRAST=0.9
BLUR=3
SATURATION=1
PYTHON_VERSION=python3
PIP_VERSION=pip3

if [ "$1" = "create" ]; then
 $PIP_VERSION install -r /content/debruits/requirements.txt
elif [[ "$1" = "preprocess" ]]; then
 $PYTHON_VERSION /content/debruits/src/preprocess.py train $BRIGHTNESS $CONTRAST $BLUR $SATURATION
 #$PYTHON_VERSION /content/debruits/src/preprocess.py val $BRIGHTNESS $CONTRAST $BLUR $SATURATION
elif [[ "$1" = "train" ]]; then
 $PYTHON_VERSION /content/debruits/src/train.py $2 $N_EPOCHS
elif [[ "$1" = "inference" ]]; then
 $PYTHON_VERSION /content/debruits/src/preprocess.py test $BRIGHTNESS $CONTRAST $BLUR $SATURATION
 $PYTHON_VERSION /content/debruits/src/inference.py
elif [[ "$1" = "crop" ]]; then
 $PYTHON_VERSION /content/debruits/src/crop.py $2
elif [[ "$1" = "super_resolution" ]]; then
 $PYTHON_VERSION /content/debruits/src/super_resolution.py $2 $N_SUPER_RESOLUTION
fi
