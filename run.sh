#!/bin/sh
N_EPOCHS=2
N_SUPER_RESOLUTION=1
BRIGHTNESS=1
CONTRAST=10
BLUR=10
SATURATION=3
PYTHON_VERSION=python3
PIP_VERSION=pip3
PREPROCESS_PY="preprocess.py"
HANDLERS_PY="handlers.py"
TRAIN_PY="train.py"
INFERENCE_PY="inference.py"
SUPER_RESOLUTION_PY="super_resolution.py"
RDN_PY="rdn.py"
IMAGE_MODEL_PY="imagemodel.py"
IMAGE_PROCESSING_PY="image_processing.py"

if [ "$1" = "create" ]; then
 $PIP_VERSION install virtualenv
 virtualenv debruitssenv
 source debruitssenv/bin/activate
 $PIP_VERSION install -r requirements.txt
 cp -t debruitssenv ./src/*
 deactivate
elif [[ "$1" = "preprocess" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $PREPROCESS_PY $BRIGHTNESS $CONTRAST $BLUR $SATURATION
 deactivate
elif [[ "$1" = "train" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $TRAIN_PY $N_EPOCHS
 deactivate
elif [[ "$1" = "inference" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $INFERENCE_PY
 deactivate
elif [[ "$1" = "super_resolution" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $SUPER_RESOLUTION_PY $N_SUPER_RESOLUTION
 deactivate
fi
