# De Bruits

Project that involves a GAN and creative art.

Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

## Setup

```
bash configure.sh
```

## Configure Jenkins

1. Open Jenkins and "Build Executor Status" > Main cluster
2. Set Number of executor=0, create a label (e.g. "kubernetes-cluster") and select "Only build jobs with labels...".
3. Install the Kubernetes plugin.
4. Create a new Cloud of type Kubernetes.
   URL is the result of "kubectl cluster-info --context kind-kind".
   Add "jenkins" as namespace.
   Select "Disable https certificate check".
   Create a new credential called "k8s-token" as long text, and paste the TOKEN value.
5. Create and run a new pipeline.