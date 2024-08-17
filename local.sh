#!/bin/bash

set -e

python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Select dataset:"
select DATASET in "rego" "art"; do
    break
done

echo "Select input filter:"
select INPUT_FILTER in "original" "solarize" "slic-10" "slic-100" "slic-1000" "color" "gaussian" "edges" "blur" "sharpen"; do
    break
done

echo "Select target filter:"
select TARGET_FILTER in "original" "solarize" "slic-10" "slic-100" "slic-1000" "color" "gaussian" "edges" "blur" "sharpen"; do
    break
done

echo "Select learning rate:"
select LEARNING_RATE in "0.01" "0.001" "0.0001"; do
    break
done

rm -rf data

if [ "$DATASET" == "rego" ]; then
    gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD
elif [ "$DATASET" == "art" ]; then
    gdown --id 1p_3PegnUsINvobi6pUF46rwVOEGc_LpC
fi
unzip data.zip && rm -rf data.zip

mkdir -p data/evolution data/model data/inference data/videos

python src/preprocess.py local $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE
python src/train.py local $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE
python src/inference.py local $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE
python src/create_video.py local $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE

rm -rf data