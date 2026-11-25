# Run — Kubernetes

**First run**

```bash
minikube start
minikube mount "$(pwd)/data:/data" &

eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .
kubectl apply -f infra/k8s/

minikube service debruits
```

**After a restart**

```bash
minikube start
minikube mount "$(pwd)/data:/data" &
kubectl apply -f infra/k8s/
minikube service debruits
```

**After code changes**

```bash
eval $(minikube docker-env)
docker build -f infra/Dockerfile -t debruits-app:latest .
kubectl rollout restart deployment/debruits
```

> `minikube mount` is required because the Docker driver runs inside a container — `hostPath` volumes are only reachable via the 9p mount.
