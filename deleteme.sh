#!/bin/bash

python src/preprocess.py local art special-input special-target 0.0001
python src/train_gan.py local art special-input special-target 0.0001