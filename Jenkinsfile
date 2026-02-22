pipeline {
    agent any

    stages {

        stage('Print Student Info') {
            steps {
                sh '''
                echo "======================================"
                echo "Name: G Sashi Preetham"
                echo "Roll No: 2022BCS0079"
                echo "======================================"
                '''
            }
        }

        stage('Create Virtual Environment') {
            steps {
                sh 'python3 -m venv venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                ./venv/bin/pip install --upgrade pip
                ./venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Run Training Script') {
            steps {
                sh '''
                ./venv/bin/python train.py
                '''
            }
        }

        stage('Print Completion Message') {
            steps {
                sh '''
                echo "======================================"
                echo "Model training completed successfully!"
                echo "Name:G Sashi Preetham"
                echo "Roll No: 2022BCS0079"
                echo "======================================"
                '''
            }
        }
    }
}
