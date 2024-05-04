#!/bin/bash

minikube start

# Clean namespace
kubectl delete all --all -n debruits
kubectl delete secrets --all -n debruits
kubectl delete configmaps --all -n debruits

# Install infrastructure using Helm templates
kubectl create namespace debruits
helm install debruits-kubernetes2024 ./debruits-kubernetes -n debruits

# Clear ports and expose MongoDB and Mongo Express
kill -9 $(lsof -t -i:8081)
kill -9 $(lsof -t -i:27017)

MONGODB_POD=$(kubectl get pods -l app=mongodb -n debruits -o jsonpath='{.items[0].metadata.name}')
MONGO_EXPRESS_POD=$(kubectl get pods -l app=mongo-express -n debruits -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward $MONGODB_POD 27017:27017 &
kubectl port-forward $MONGO_EXPRESS_POD 8081:8081 &
#password mongo-express: admin/pass

sudo rm -rf venv
sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python3.10 "./src/encode_images.py"
python3.10 "./src/preprocess.py"
python3.10 "./src/train.py"
python3.10 "./src/inference.py"


python3.10 "./src/super_resolution.py"
python3.10 "./src/create_video.py"