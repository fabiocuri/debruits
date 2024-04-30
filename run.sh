#!/bin/bash

minikube start
kubectl delete all --all -n debruits

export MONGODB_USERNAME="debruits"
export MONGODB_USERNAME_BASE64=echo -n $MONGODB_USERNAME | base64
export MONGODB_PASSWORD="debruits"
export MONGODB_PASSWORD_BASE64=echo -n $MONGODB_PASSWORD | base64

kubectl apply -f ./kubernetes/mongodb-configmap.yaml
kubectl apply -f ./kubernetes/mongodb-secrets.yaml
kubectl apply -f ./kubernetes/mongodb.yaml
kubectl apply -f ./kubernetes/mongodb-express.yaml
kubectl port-forward mongodb-7487f5894b-zm4hk 27017:27017 &
kubectl port-forward mongo-express-86cf48bd49-shnxn 8081:8081 &

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