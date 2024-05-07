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
envsubst < kubernetes/cluster.yaml | kind create cluster --retain --config=-
kubectl cluster-info --context kind-kind

echo "--------------------------------------------"
echo "Setting up Jenkins..."
echo "--------------------------------------------"
kubectl create namespace jenkins
kubens jenkins
helm repo add jenkins https://charts.jenkins.io
helm repo update
helm install jenkins jenkins/jenkins
kubectl apply -f kubernetes/jenkins-token.yaml
echo "-----------BEGINNING TOKEN-----------"
kubectl describe secret $(kubectl describe serviceaccount jenkins | grep token | awk '{print $2}')
echo "-----------END TOKEN-----------"
kubectl create rolebinding jenkins-admin-binding --clusterrole=admin --serviceaccount=jenkins:jenkins

echo "--------------------------------------------"
echo "Setting up MongoDB..."
echo "--------------------------------------------"
sleep 20
kubectl apply -f kubernetes/debruits-configmap.yaml
kubectl apply -f kubernetes/debruits-secret.yaml
kubectl apply -f kubernetes/mongodb.yaml
export CLUSTER_NODE_ID=$(kubectl get node -o wide | awk 'NR==2 {print $6}')
cp config.yaml config_pipeline.yaml
sed -i "s/\$CLUSTER_NODE_ID/$cluster_node_id/" config_pipeline.yaml

echo "--------------------------------------------"
echo "Setting up Mongo Express..."
echo "--------------------------------------------"
kubectl apply -f kubernetes/mongodb-express.yaml

echo "--------------------------------------------"
echo "Exposing ports..."
echo "--------------------------------------------"
sleep 90
export JENKINS_POD=$(kubectl get pods -l app.kubernetes.io/name=jenkins -o jsonpath='{.items[0].metadata.name}')
export MONGODB_POD=$(kubectl get pods -l app=mongodb -o jsonpath='{.items[0].metadata.name}')
export MONGO_EXPRESS_POD=$(kubectl get pods -l app=mongo-express -o jsonpath='{.items[0].metadata.name}')
{
  kill -9 $(lsof -t -i:8080) || true
  kill -9 $(lsof -t -i:27017) || true
  kill -9 $(lsof -t -i:8081) || true
} &>/dev/null
kubectl port-forward $JENKINS_POD 8080:8080 &
kubectl port-forward $MONGODB_POD 27017:27017 &
kubectl port-forward $MONGO_EXPRESS_POD 8081:8081 &
echo "You can access Jenkins through https://localhost:8080" #admin/[kubectl exec -it svc/jenkins bash][cat /run/secrets/additional/chart-admin-password]
echo "You can access MongoDB through https://localhost:27017 and https://${CLUSTER_NODE_ID}:30000" #admin/pass
echo "You can access Mongo Express through https://localhost:8081" #admin/pass

echo "--------------------------------------------"
echo "Downloading data..."
echo "--------------------------------------------"
gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD
unzip data.zip
rm -rf data.zip