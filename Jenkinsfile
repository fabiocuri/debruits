pipeline {
  agent {
    kubernetes {
      yaml '''
        apiVersion: v1
        kind: Pod
        spec:
          containers:
          - name: python
            image: python:latest
            command:
            - cat
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
    stage('download-data') {
      steps {
          sh 'gdown --id 1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD'
          sh 'unzip data.zip'
          sh 'rm -rf data.zip'
      }
    }
  }
}