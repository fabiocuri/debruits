pipeline {
    agent any
    environment {
        PYTHON_VERSION = 'python3.10'
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