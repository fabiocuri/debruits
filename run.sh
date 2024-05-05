#!/bin/bash

# Install and configure Jenkins

docker run -p 8080:8080 -p 50000:50000 -d -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkins/jenkins:lts
docker exec -u 0 -it 5f4841c68399 bash

# Install Docker
curl https://get.docker.com/ > dockerinstall && chmod 777 dockerinstall && ./dockerinstall
chmod 666 /var/run/docker.sock

# Install pip and Python
apt-get update
apt-get install -y python3 python3-pip

# Install kubectl
curl -LO https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl; chmod +x ./kubectl; mv ./kubectl /usr/local/bin/kubectl



minikube start

# Clean namespace
kubectl delete all --all -n debruits
kubectl delete secrets --all -n debruits
kubectl delete configmaps --all -n debruits

# Install infrastructure using Helm templates
kubectl create namespace debruits
helm repo add jenkins https://charts.jenkins.io
helm repo update
helm install debruits-kubernetes ./debruits-kubernetes -n debruits
helm install jenkins jenkins/jenkins

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