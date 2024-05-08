pipeline {
  agent {
    kubernetes {
      yaml '''
        apiVersion: v1
        kind: Pod
        spec:
          containers:
          - name: python
            image: python:3.10.12
            command:
            - /bin/bash
            - -c
            - |
              apt-get update && apt-get install -y python3-opencv && pip install opencv-python
            args:
            - "cat"
            tty: true
        '''
    }
  }
  stages {
    stage('install-requirements') {
      steps {
        container('python') {
          sh 'pip install -r requirements.txt'
        }
      }
    }
    stage('data-preprocess') {
      steps {
        container('python') {
          sh 'python ./src/preprocess.py'
        }
      }
    }
    stage('model-train') {
      steps {
        container('python') {
          sh 'python ./src/train.py'
        }
      }
    }
    stage('model-inference') {
      steps {
        container('python') {
          sh 'python ./src/inference.py'
        }
      }
    }
  }
}