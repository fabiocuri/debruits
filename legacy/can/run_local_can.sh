#!/bin/bash

python3.7 -m venv venv
source venv/bin/activate
pip3.7 install -r requirements_can.txt

python3.7 src/main.py