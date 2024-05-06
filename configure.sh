#!/bin/bash

set -e

echo "--------------------------------------------"
echo "Installing docker..."
echo "--------------------------------------------"

sudo apt install docker.io

echo "--------------------------------------------"
echo "Installing kubectl..."
echo "--------------------------------------------"
snap install kubectl --classic

echo "--------------------------------------------"
echo "Installing kind..."
echo "--------------------------------------------"
[ $(uname -m) = x86_64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.22.0/kind-linux-amd64
[ $(uname -m) = aarch64 ] && curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.22.0/kind-linux-arm64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

echo "--------------------------------------------"
echo "Setting up kubernetes cluster..."
echo "--------------------------------------------"
ip_addresses=$(hostname -I)
export MY_IP_ADDRESS=$(echo "$ip_addresses" | awk '{print $1}')
kind delete cluster --name kind
envsubst < ./kubernetes/cluster.yaml | kind create cluster --retain --config=-
kubectl cluster-info --context kind-kind

echo "--------------------------------------------"
echo "Setting up Jenkins..."
echo "--------------------------------------------"
kubectl create namespace jenkins
kubens jenkins
helm repo add jenkins https://charts.jenkins.io
helm repo update
helm install jenkins jenkins/jenkins
kubectl apply -f ./kubernetes/jenkins-token.yaml
echo "-----------BEGINNING TOKEN-----------"
kubectl describe secret $(kubectl describe serviceaccount jenkins | grep token | awk '{print $2}')
echo "-----------END TOKEN-----------"
kubectl create rolebinding jenkins-admin-binding --clusterrole=admin --serviceaccount=jenkins:jenkins

echo "--------------------------------------------"
echo "Setting up MongoDB..."
echo "--------------------------------------------"
kubectl apply -f ./kubernetes/debruits-configmap.yaml
kubectl apply -f ./kubernetes/debruits-secret.yaml
kubectl apply -f ./kubernetes/mongodb.yaml

echo "--------------------------------------------"
echo "Setting up Mongo Express..."
echo "--------------------------------------------"
kubectl apply -f ./kubernetes/mongodb-express.yaml

echo "--------------------------------------------"
echo "Exposing ports..."
echo "--------------------------------------------"
kubectl port-forward jenkins-0 8080:8080 &
kubectl port-forward jenkins-0 8081:8081 &
echo "You can access Jenkins through https://localhost:8080"
echo "You can access Mongo Express through https://localhost:8081"

# #!/bin/bash

# # Clear ports and expose MongoDB and Mongo Express
# kill -9 $(lsof -t -i:8080)
# kill -9 $(lsof -t -i:8081)
# kill -9 $(lsof -t -i:27017)

# JENKINS_POD=$(kubectl get pods -l app.kubernetes.io/name=jenkins -n debruits  -o jsonpath='{.items[0].metadata.name}')
# MONGODB_POD=$(kubectl get pods -l app=mongodb -n debruits -o jsonpath='{.items[0].metadata.name}')
# MONGO_EXPRESS_POD=$(kubectl get pods -l app=mongo-express -n debruits -o jsonpath='{.items[0].metadata.name}')
# kubectl port-forward $JENKINS_POD 8080:8080 &
# kubectl port-forward $MONGODB_POD 27017:27017 &
# kubectl port-forward $MONGO_EXPRESS_POD 8081:8081 &
# #password mongo-express: admin/pass

# sudo rm -rf venv
# sudo apt install python3.10-venv
# python3.10 -m venv venv
# source venv/bin/activate

#                     sh 'gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD'
#                     sh 'unzip data.zip'

# pip install -r requirements.txt

# python3.10 "./src/encode_images.py"
# python3.10 "./src/preprocess.py"
# python3.10 "./src/train.py"
# python3.10 "./src/inference.py"


# python3.10 "./src/super_resolution.py"
# python3.10 "./src/create_video.py"