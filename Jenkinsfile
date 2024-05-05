#!/usr/bin/env groovy
pipeline {
    agent any
    stages {
        stage('retrieve-data') {
            steps {
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
        stage('commit version update') {
            steps {
                python3.10 "./src/inference.py"
            }
        }
    }
}