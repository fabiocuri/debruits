#!/bin/sh
PYTHON_VERSION=python3
PIP_VERSION=pip3
PREPROCESS_PY="preprocess.py"
HANDLERS_PY="handlers.py"
TRAIN_PY="train.py"
INFERENCE_PY="inference.py"
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
fi
