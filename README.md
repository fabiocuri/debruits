# De Bruits

De Bruits is an innovative project that merges the realms of Generative Adversarial Networks (GANs) and creative art.

**Author:** Fabio Curi  
**Contact:** fcuri91@gmail.com  

![Image Alt Text](image/Expo_7.png)

## Stack

Language: Python, Bash
CI/CD: Jenkins, Github
Database: MongoDB
Deployment: Kubernetes

## Setup

To set up the project, run the following command in your terminal:

```bash
bash configure.sh
```

## Jenkins Configuration

```markdown
Configure Jenkins for seamless integration with the project:

1. Open Jenkins and navigate to "Build Executor Status" > Main cluster.
2. Set the Number of executors to 0 and create a label (e.g., "kubernetes-cluster"). Select "Only build jobs with labels...".
3. Install the Kubernetes plugin.
4. Create a new Cloud of type Kubernetes. Use the URL obtained from `kubectl cluster-info --context kind-kind`.
   - Add "jenkins" as the namespace.
   - Select "Disable https certificate check".
   - Create a new credential named "k8s-token" as a long text, and paste the TOKEN value.
5. Create and run a new pipeline pointing to this project.
6. Install "Stage View" plugin for better visualisation of pipeline.
