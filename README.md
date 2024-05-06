# De Bruits

Project that involves a GAN and creative art.

Author: Fabio Curi Paixao 

E-mail: fcuri91@gmail.com

## Setup

```
bash configure.sh
```

## Configure Jenkins

1. Go to Jenkins, click on "Build Executor Status", then the main cluster, and set number of executors to 0, and create a label (e.g. "kubernetes").
2. Install the Kubernetes plugin.
3. Open "Manage Jenkins", "Clouds", "New cloud" and create a "kubernetes" Cloud of type Kubernetes.
   Add the URL that is the result of "kubectl cluster-info --context kind-kind".
   Add "jenkins" as namespace.
   Click on "Disable https certificate check".
   Create a new credential called "k8s-token" as long text, and paste the TOKEN value.
   Select "Websocket" and click on save.

## Create Pipeline in Jenkins