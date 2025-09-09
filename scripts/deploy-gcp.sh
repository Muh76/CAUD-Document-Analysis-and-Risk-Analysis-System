#!/bin/bash

# Google Cloud Run Deployment Script
set -e

PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME=${GCP_SERVICE_NAME:-"contract-analysis-api"}
REGION=${GCP_REGION:-"us-central1"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "🚀 Deploying to Google Cloud Run..."
echo "Project: ${PROJECT_ID}"
echo "Service: ${SERVICE_NAME}"
echo "Region: ${REGION}"

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -f docker/api.Dockerfile -t ${IMAGE_NAME} .

echo "⬆️ Pushing to Google Container Registry..."
docker push ${IMAGE_NAME}

# Deploy to Cloud Run
echo "�� Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars ENVIRONMENT=production,LOG_LEVEL=INFO

echo "✅ Deployment complete!"
echo "🌐 Service URL: https://${SERVICE_NAME}-${PROJECT_ID}.a.run.app"
