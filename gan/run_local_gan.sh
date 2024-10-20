#!/bin/bash

python -m venv venv
source venv/bin/activate
pip install -r requirements_gan.txt

python src/preprocess.py local art special-input original 0.01
python src/train_gan.py local art special-input original 0.01
python src/superresolution.py data/evolution_gan
python src/create_video.py local data/super_resolution x x x x