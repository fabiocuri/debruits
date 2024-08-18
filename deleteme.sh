#!/bin/bash

python src/preprocess.py local art special-input special-target 0.0001
python src/train_gan.py local art special-input special-target 0.0001
python src/superresolution.py data/sequence_borosylicate
python src/create_video.py local data/super_resolution x x x x