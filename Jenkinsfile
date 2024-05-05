#!/usr/bin/env groovy
pipeline {
    agent any
    stages {
        stage('retrieve-data') {
            steps {
                pip install -r requirements.txt
                gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD
                python3.10 "./src/encode_images.py"
            }
        }
        stage('preprocess-data') {
            steps {
                python3.10 "./src/preprocess.py"
            }
        }
        stage('train-model') {
            steps {
                python3.10 "./src/train.py"
            }
        }
        stage('inference-model') {
            steps {
                python3.10 "./src/inference.py"
            }
        }
    }
}