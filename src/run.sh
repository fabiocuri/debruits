#!/bin/sh
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
 cd envs
 virtualenv debruitssenv
 source debruitssenv/bin/activate
 $PIP_VERSION install -r requirements.txt
 deactivate
elif [[ "$1" = "preprocess" ]]; then
 cp -t envs/debruitssenv $PREPROCESS_PY $HANDLERS_PY
 cd envs/debruitssenv
 source bin/activate
 $PYTHON_VERSION $PREPROCESS_PY
 deactivate
elif [[ "$1" = "train" ]]; then
 cp -t envs/debruitssenv $TRAIN_PY
 cd envs/debruitssenv
 source bin/activate
 $PYTHON_VERSION $TRAIN_PY
 deactivate
elif [[ "$1" = "inference" ]]; then
 cp -t envs/debruitssenv $INFERENCE_PY
 cd envs/debruitssenv
 source bin/activate
 $PYTHON_VERSION $INFERENCE_PY
 deactivate
elif [[ "$1" = "super_resolution" ]]; then
 cp -t envs/debruitssenv $SUPER_RESOLUTION_PY $RDN_PY $IMAGE_MODEL_PY $IMAGE_PROCESSING_PY
 cd envs/debruitssenv
 source bin/activate
 $PYTHON_VERSION $SUPER_RESOLUTION_PY "output1.png"
 deactivate
fi
