#!/bin/bash

minikube start

# Clean namespace
kubectl delete all --all -n debruits

# Install infrastructure using Helm templates
helm install debruits-kubernetes ./debruits-kubernetes

# Clear ports and expose MongoDB and Mongo Express
kill -9 $(lsof -t -i:8081)
kill -9 $(lsof -t -i:27017)

MONGODB_POD=$(kubectl get pods -l app=mongodb -o jsonpath='{.items[0].metadata.name}')
MONGO_EXPRESS_POD=$(kubectl get pods -l app=mongo-express -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward $MONGODB_POD 27017:27017 &
kubectl port-forward $MONGO_EXPRESS_POD 8081:8081 &

sudo rm -rf venv
sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

SRC_PATH=./src

clear

python3.10 "$SRC_PATH/encode_images.py"

python3.10 "$SRC_PATH/train.py"
python3.10 "$SRC_PATH/inference.py"
python3.10 "$SRC_PATH/super_resolution.py"
python3.10 "$SRC_PATH/create_video.py"