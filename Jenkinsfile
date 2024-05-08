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
            command: ["cat"]
            args: []
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
        input message: 'Please select parameters for data preprocessing', parameters: [
          choice(name: 'INPUT_FILTER', choices: ['original', 'solarize', 'slic-10', 'slic-100', 'slic-1000', 'color', 'gaussian', 'edges', 'blur', 'sharpen'], description: 'Select input filter'),
          choice(name: 'TARGET_FILTER', choices: ['original', 'solarize', 'slic-10', 'slic-100', 'slic-1000', 'color', 'gaussian', 'edges', 'blur', 'sharpen'], description: 'Select target filter'),
          choice(name: 'LEARNING_RATE', choices: ['0.01', '0.001', '0.0001'], description: 'Learning rate')
        ]
        container('python') {
          script {
            def INPUT_FILTER = params.INPUT_FILTER
            def TARGET_FILTER = params.TARGET_FILTER
            def LEARNING_RATE = params.LEARNING_RATE
            sh "python ./src/preprocess.py $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
          }
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