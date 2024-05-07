pipeline {
  agent {
    kubernetes {
      yaml '''
        apiVersion: v1
        kind: Pod
        spec:
          containers:
          - name: python
            image: python:alpine
            command:
            - cat
            tty: true
        '''
    }
  }
  stages {
    stage('Install Python packages') {
      steps {
        container('python') {
          sh 'pip install -r requirements.txt'
        }
      }
    }
  }
}