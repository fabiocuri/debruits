#!/bin/sh
N_EPOCHS=100
N_SUPER_RESOLUTION=2
BRIGHTNESS=10
CONTRAST=1
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
CROP_PY="crop.py"
SUPER_RESOLUTION_FOLDER="../../debruits-images/sequence1/cropped"
TO_BE_CROPPED_FOLDER="../../debruits-images/sequence1"

if [ "$1" = "create" ]; then
 $PIP_VERSION install -r /content/debruits/requirements.txt
elif [[ "$1" = "preprocess" ]]; then
 $PYTHON_VERSION /content/debruits/src/$PREPROCESS_PY $BRIGHTNESS $CONTRAST $BLUR $SATURATION
elif [[ "$1" = "train" ]]; then
 $PYTHON_VERSION /content/debruits/src/$TRAIN_PY $N_EPOCHS
elif [[ "$1" = "inference" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $INFERENCE_PY
 deactivate
elif [[ "$1" = "super_resolution" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $SUPER_RESOLUTION_PY $N_SUPER_RESOLUTION $SUPER_RESOLUTION_FOLDER
 deactivate
elif [[ "$1" = "crop" ]]; then
 cd debruitssenv
 source bin/activate
 $PYTHON_VERSION $CROP_PY $TO_BE_CROPPED_FOLDER
 deactivate
fi
