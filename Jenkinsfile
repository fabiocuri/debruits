pipeline {
    agent {
        kubernetes {
            label 'cluster-label'
        }
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