pipeline {
    agent any
    environment {
        PYTHON_VERSION = 'python3.10'
    }
    stages {
        stage('setup') {
            steps {
                script {
                    sh 'sudo apt update'
                    sh 'sudo apt install python3-pip'
                    sh 'pip install -r requirements.txt'
                    sh 'gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD'
                    sh "${PYTHON_VERSION} ./src/encode_images.py"
                }
            }
        }
        stage('preprocess-data') {
            steps {
                script {
                    sh "${PYTHON_VERSION} ./src/preprocess.py"
                }
            }
        }
        stage('train-model') {
            steps {
                script {
                    sh "${PYTHON_VERSION} ./src/train.py"
                }
            }
        }
        stage('inference-model') {
            steps {
                script {
                    sh "${PYTHON_VERSION} ./src/inference.py"
                }
            }
        }
    }
}