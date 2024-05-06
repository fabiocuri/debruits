pipeline {
    agent {
        kubernetes 
    }
    stages {
        stage('setup-k8s') {
            steps {
                script {
                    sh 'kubectl create namespace debruits'
                }
            }
        }
    }
}