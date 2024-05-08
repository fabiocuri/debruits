# De Bruits

De Bruits is an innovative project that merges the realms of Generative Adversarial Networks (GANs) and creative art.

**Author:** Fabio Curi  
**Contact:** fcuri91@gmail.com  

![Image Alt Text](image/Expo_7.png)

## Stack

Language: Python, Bash, Groovy

CI/CD: Jenkins, Github

Database: MongoDB

Deployment: Kubernetes

Messaging: Sendgrid

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
     TOKEN is result from `kubectl describe secret $(kubectl describe serviceaccount jenkins | grep token | awk '{print $2}')`.
5. Create and run a new pipeline pointing to this project.
6. Install "Stage View" plugin for better visualisation of pipeline.

## Sendgrid Configuration

1. Create an account in Sendgrid.
2. Create a SMTP integration.
3. Create a user in Sendgrid, which will be used to send the e-mails.
4. Open Jenkins and install the "Extend Email" plugin.
5. Go to "System" and "Extended E-mail Notification".
   - Set "SMTP server" to "smtp.sendgrid.net".
   - Set "SMTP Port" to 465.
   - Create Sendgrid credentials.
   - Select "Use SSL".