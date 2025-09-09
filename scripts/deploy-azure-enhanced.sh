#!/bin/bash

# Enhanced Azure Container Apps Deployment Script
set -e

# Configuration
RESOURCE_GROUP=${AZURE_RESOURCE_GROUP:-"contract-analysis-rg"}
LOCATION=${AZURE_LOCATION:-"East US"}
CONTAINER_APP_NAME=${AZURE_CONTAINER_APP_NAME:-"contract-analysis"}
ENVIRONMENT=${DEPLOYMENT_ENVIRONMENT:-"production"}
REGISTRY_NAME="${CONTAINER_APP_NAME}registry"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Deploying Contract Analysis System to Azure Container Apps${NC}"
echo -e "${YELLOW}Resource Group: ${RESOURCE_GROUP}${NC}"
echo -e "${YELLOW}Container App: ${CONTAINER_APP_NAME}${NC}"
echo -e "${YELLOW}Location: ${LOCATION}${NC}"
echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"

# Check if logged in to Azure
if ! az account show &> /dev/null; then
    echo -e "${RED}❌ Error: Please login to Azure CLI first${NC}"
    echo "Run: az login"
    exit 1
fi

# Create resource group if it doesn't exist
echo -e "${GREEN}📁 Creating resource group...${NC}"
az group create --name ${RESOURCE_GROUP} --location "${LOCATION}" || echo "Resource group already exists"

# Create container registry
echo -e "${GREEN}📦 Creating container registry...${NC}"
az acr create --resource-group ${RESOURCE_GROUP} \
    --name ${REGISTRY_NAME} \
    --sku Basic \
    --admin-enabled true || echo "Registry already exists"

# Build and push image
echo -e "${GREEN}🔨 Building and pushing image...${NC}"
az acr build --registry ${REGISTRY_NAME} \
    --image ${CONTAINER_APP_NAME}:latest \
    --file docker/api.Dockerfile .

# Create container app environment
echo -e "${GREEN}🌍 Creating container app environment...${NC}"
az containerapp env create --name ${CONTAINER_APP_NAME}-env \
    --resource-group ${RESOURCE_GROUP} \
    --location "${LOCATION}" || echo "Environment already exists"

# Deploy container app
echo -e "${GREEN}🚀 Deploying container app...${NC}"
az containerapp create --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --environment ${CONTAINER_APP_NAME}-env \
    --image ${REGISTRY_NAME}.azurecr.io/${CONTAINER_APP_NAME}:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 10 \
    --cpu 2.0 \
    --memory 4.0Gi \
    --env-vars ENVIRONMENT=${ENVIRONMENT} LOG_LEVEL=INFO PROMETHEUS_ENABLED=true || \
    az containerapp update --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --image ${REGISTRY_NAME}.azurecr.io/${CONTAINER_APP_NAME}:latest

# Get service URL
SERVICE_URL=$(az containerapp show --name ${CONTAINER_APP_NAME} --resource-group ${RESOURCE_GROUP} --query "properties.configuration.ingress.fqdn" -o tsv)

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}🌐 Service URL: https://${SERVICE_URL}${NC}"
echo -e "${GREEN}📊 Health Check: https://${SERVICE_URL}/health${NC}"
echo -e "${GREEN}📈 Metrics: https://${SERVICE_URL}/metrics${NC}"

# Test deployment
echo -e "${GREEN}�� Testing deployment...${NC}"
curl -f https://${SERVICE_URL}/health || echo -e "${RED}❌ Health check failed${NC}"
