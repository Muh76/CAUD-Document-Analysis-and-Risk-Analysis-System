#!/bin/bash

# Enhanced Google Cloud Run Deployment Script
set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"arched-catwalk-459814-b0"}
SERVICE_NAME=${GCP_SERVICE_NAME:-"contract-analysis-api"}
REGION=${GCP_REGION:-"us-central1"}
ENVIRONMENT=${DEPLOYMENT_ENVIRONMENT:-"production"}
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Contract Analysis System to Google Cloud Run${NC}"
echo -e "${YELLOW}Project: ${PROJECT_ID}${NC}"
echo -e "${YELLOW}Service: ${SERVICE_NAME}${NC}"
echo -e "${YELLOW}Region: ${REGION}${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Validate configuration
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo -e "${RED}❌ Error: Please set GCP_PROJECT_ID environment variable${NC}"
    exit 1
fi

# Set project
echo -e "${GREEN}📁 Setting GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "${GREEN}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and push Docker image using Cloud Build
echo -e "${GREEN}📦 Building Docker image with Cloud Build...${NC}"
gcloud builds submit --tag ${IMAGE_NAME}:latest .

# Deploy to Cloud Run
echo -e "${GREEN}🚀 Deploying to Cloud Run...${NC}"
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars ENVIRONMENT=${ENVIRONMENT},LOG_LEVEL=INFO,PROMETHEUS_ENABLED=true \
    --timeout 300 \
    --concurrency 100 \
    --max-instances 10

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}🌐 Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}📊 Health Check: ${SERVICE_URL}/health${NC}"
echo -e "${GREEN}�� Metrics: ${SERVICE_URL}/metrics${NC}"

# Test deployment
echo -e "${GREEN}�� Testing deployment...${NC}"
curl -f ${SERVICE_URL}/health || echo -e "${RED}❌ Health check failed${NC}"
