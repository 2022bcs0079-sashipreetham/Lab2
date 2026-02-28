pipeline {
    agent any

    environment {
        IMAGE_NAME = "wine-quality-api:latest"
        CONTAINER_NAME = "wine_test_container"
        INTERNAL_PORT = "8000"
    }

    stages {

        // -----------------------------
        // Stage 1: Pull Image
        // -----------------------------
        stage('Pull Image') {
            steps {
                sh '''
                echo "Pulling Docker image..."
                docker pull $IMAGE_NAME
                '''
            }
        }

        // -----------------------------
        // Stage 2: Run Container
        // -----------------------------
        stage('Run Container') {
            steps {
                sh '''
                echo "Removing old container if exists..."
                docker rm -f $CONTAINER_NAME || true
                
                echo "Starting container..."
                docker run -d --name $CONTAINER_NAME \
                  -v ${WORKSPACE}/tests:/tests \
                  $IMAGE_NAME
                '''
            }
        }

        // -----------------------------
        // Stage 3: Wait for Service Readiness
        // -----------------------------
        stage('Wait for Service Readiness') {
            steps {
                sh '''
                echo "Waiting for API to be ready..."
                for i in {1..20}
                do
                  sleep 2
                  STATUS=$(docker exec $CONTAINER_NAME \
                    curl -s -o /dev/null -w "%{http_code}" \
                    http://localhost:$INTERNAL_PORT/health || true)

                  if [ "$STATUS" = "200" ]; then
                    echo "Service is ready!"
                    exit 0
                  fi
                done
                echo "Service did not start in time."
                exit 1
                '''
            }
        }

        // -----------------------------
        // Debug: Container Files
        // -----------------------------
        stage('Debug Container Files') {
            steps {
                sh '''
                echo "Listing container root..."
                docker exec $CONTAINER_NAME ls /

                echo "Listing /tests directory..."
                docker exec $CONTAINER_NAME ls /tests || true
                '''
            }
        }

        // -----------------------------
        // Debug: Workspace
        // -----------------------------
        stage('Debug Workspace') {
            steps {
                sh '''
                echo "Workspace path: $WORKSPACE"
                ls -R $WORKSPACE
                '''
            }
        }

        // -----------------------------
        // Stage 4: Valid Inference Test
        // -----------------------------
        stage('Send Valid Inference Request') {
            steps {
                sh '''
                echo "Sending valid request..."
                RESPONSE=$(cat tests/valid_input.json | docker exec -i $CONTAINER_NAME sh -c '
                  curl -s -w "\\n%{http_code}" -X POST \
                  http://localhost:8000/predict \
                  -H "Content-Type: application/json" \
                  -d @-
                ')
                BODY=$(echo "$RESPONSE" | head -n 1)
                STATUS=$(echo "$RESPONSE" | tail -n 1)

                echo "Status Code: $STATUS"
                echo "Response Body: $BODY"

                if [ "$STATUS" != "200" ]; then
                  echo "Valid request failed!"
                  exit 1
                fi

                echo "$BODY" | grep -q "prediction" || {
                  echo "Prediction field missing!"
                  exit 1
                }

                echo "Valid inference test passed."
                '''
            }
        }

        // -----------------------------
        // Stage 5: Invalid Request Test
        // -----------------------------
        stage('Send Invalid Request') {
            steps {
                sh '''
                echo "Sending invalid request..."
                RESPONSE=$(cat tests/invalid_input.json | docker exec -i $CONTAINER_NAME sh -c '
                  curl -s -w "\\n%{http_code}" -X POST \
                  http://localhost:8000/predict \
                  -H "Content-Type: application/json" \
                  -d @-
                ')
                BODY=$(echo "$RESPONSE" | head -n 1)
                STATUS=$(echo "$RESPONSE" | tail -n 1)

                echo "Status Code: $STATUS"
                echo "Response Body: $BODY"

                if [ "$STATUS" = "200" ]; then
                  echo "Invalid request unexpectedly succeeded!"
                  exit 1
                fi

                echo "Invalid request test passed."
                '''
            }
        }

        // -----------------------------
        // Stage 6: Stop Container
        // -----------------------------
        stage('Stop Container') {
            steps {
                sh '''
                echo "Stopping and removing container..."
                docker stop $CONTAINER_NAME || true
                docker rm $CONTAINER_NAME || true
                '''
            }
        }
    }

    // -----------------------------
    // Stage 7: Final Result
    // -----------------------------
    post {
        success {
            echo "All validation tests passed. Pipeline SUCCESS. 2022BCS0028"
        }
        failure {
            echo "Pipeline FAILED due to validation error. 2022BCS0028"
        }
        always {
            sh '''
            echo "Ensuring container cleanup..."
            docker stop $CONTAINER_NAME || true
            docker rm $CONTAINER_NAME || true
            '''
        }
    }
}
