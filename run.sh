#!/bin/bash

# Connect to remote server
ssh -i ~/.ssh/id_rsa.pub root@167.71.47.83

# Install general dependencies
apt install docker.io
snap install kubectl --classic
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube_latest_amd64.deb
sudo dpkg -i minikube_latest_amd64.deb
rm minikube_latest_amd64.deb
minikube start --driver=docker --force

# Install and configure Jenkins
docker run -p 8080:8080 -p 50000:50000 -d -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkins/jenkins:lts
JENKINS_CONTAINER_ID=$(docker ps | grep jenkins | awk '{print $1}')
docker exec -u 0 -it $JENKINS_CONTAINER_ID bash
curl https://get.docker.com/ > dockerinstall && chmod 777 dockerinstall && ./dockerinstall
chmod 666 /var/run/docker.sock

## Install pip and Python
apt-get update
apt-get install -y python3 python3-pip

## Install kubectl
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl; chmod +x ./kubectl; mv ./kubectl /usr/local/bin/kubectl




# Clear ports and expose MongoDB and Mongo Express
kill -9 $(lsof -t -i:8080)
kill -9 $(lsof -t -i:8081)
kill -9 $(lsof -t -i:27017)

JENKINS_POD=$(kubectl get pods -l app.kubernetes.io/name=jenkins -n debruits  -o jsonpath='{.items[0].metadata.name}')
MONGODB_POD=$(kubectl get pods -l app=mongodb -n debruits -o jsonpath='{.items[0].metadata.name}')
MONGO_EXPRESS_POD=$(kubectl get pods -l app=mongo-express -n debruits -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward $JENKINS_POD 8080:8080 &
kubectl port-forward $MONGODB_POD 27017:27017 &
kubectl port-forward $MONGO_EXPRESS_POD 8081:8081 &
#password mongo-express: admin/pass
#password jenkins: admin/pass

sudo rm -rf venv
sudo apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

wget 
python3.10 "./src/encode_images.py"
python3.10 "./src/preprocess.py"
python3.10 "./src/train.py"
python3.10 "./src/inference.py"


python3.10 "./src/super_resolution.py"
python3.10 "./src/create_video.py"