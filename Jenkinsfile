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
  parameters {
    choice(name: 'DATASET', choices: ['rego', 'parque'], description: 'Select dataset')
    choice(name: 'INPUT_FILTER', choices: ['original', 'solarize', 'slic-10', 'slic-100', 'slic-1000', 'color', 'gaussian', 'edges', 'blur', 'sharpen'], description: 'Select input filter')
    choice(name: 'TARGET_FILTER', choices: ['original', 'solarize', 'slic-10', 'slic-100', 'slic-1000', 'color', 'gaussian', 'edges', 'blur', 'sharpen'], description: 'Select target filter')
    choice(name: 'LEARNING_RATE', choices: ['0.01', '0.001', '0.0001'], description: 'Learning rate')
  }
  stages {
    stage('install-requirements') {
      steps {
        container('python') {
          sh 'pip install -r requirements.txt'
        }
      }
    }
    stage('data-download') {
      steps {
        container('python') {
          script {
            def DATASET = params.DATASET
            if (params.DATASET == 'rego') {
              def DATASET_PATH = '1BPJQ1pRoCnUxYWP65Xklufgtl85kg1dD'
            } else if (params.DATASET == 'parque') {
              def DATASET_PATH = '1NqL8zJGZO7FrBKe7NKlY1YLBsxUJdSGY'
            }
          }
          sh "gdown --id $DATASET_PATH"
          sh 'unzip data.zip && rm -rf data.zip'
        }
      }
    }
    stage('data-encode') {
      steps {
        container('python') {
          sh "python src/encode_images.py $DATASET"
        }
      }
    }
    stage('data-preprocess') {
      steps {
        container('python') {
          script {
            def INPUT_FILTER = params.INPUT_FILTER
            def TARGET_FILTER = params.TARGET_FILTER
            def LEARNING_RATE = params.LEARNING_RATE
            sh "python src/preprocess.py $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
          }
        }
      }
    }
    stage('model-train') {
      steps {
        container('python') {
          sh "python src/train.py $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
        }
      }
    }
    stage('model-inference') {
      steps {
        container('python') {
          sh "python src/inference.py $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
        }
      }
    }
    stage('super-resolution') {
      steps {
        container('python') {
          sh "python src/super_resolution.py $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
        }
      }
    }
    stage('create-video') {
      steps {
        container('python') {
          sh "python src/create_video.py $DATASET $INPUT_FILTER $TARGET_FILTER $LEARNING_RATE"
          sh "mv *.mp4 videos/"
        }
      }
    }
    stage('send-email') {
      steps {
        script {
          emailext(
            to: 'fcuri91@gmail.com',
            subject: "De Bruits MP4 ($DATASET-$INPUT_FILTER-$TARGET_FILTER-$LEARNING_RATE)",
            body: 'Please find the attached videos generated by Jenkins.',
            attachmentsPattern: 'videos/*.mp4'
          )
        }
      }
    }
  }
}