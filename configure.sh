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
envsubst < cluster.yaml | kind create cluster --retain --config=-
kubectl cluster-info --context kind-kind
kubectl create namespace jenkins
kubectl create serviceaccount jenkins --namespace=jenkins
kubectl apply -f jenkins-token.yaml --namespace jenkins
echo "-----------BEGINNING TOKEN-----------"
kubectl describe secret jenkins-token --namespace jenkins
echo "-----------END TOKEN-----------"
kubectl create rolebinding jenkins-admin-binding --clusterrole=admin --serviceaccount=jenkins:jenkins --namespace=jenkins