#!/bin/bash

# Azure Container Apps Deployment Script
set -e

RESOURCE_GROUP=${AZURE_RESOURCE_GROUP:-"contract-analysis-rg"}
LOCATION=${AZURE_LOCATION:-"East US"}
CONTAINER_APP_NAME=${AZURE_CONTAINER_APP_NAME:-"contract-analysis"}
REGISTRY_NAME=${AZURE_REGISTRY_NAME:-"contractanalysisregistry"}

echo "🚀 Deploying to Azure Container Apps..."
echo "Resource Group: ${RESOURCE_GROUP}"
echo "Container App: ${CONTAINER_APP_NAME}"
echo "Location: ${LOCATION}"

# Create resource group if it doesn't exist
echo "📁 Creating resource group..."
az group create --name ${RESOURCE_GROUP} --location "${LOCATION}"

# Create container registry
echo "📦 Creating container registry..."
az acr create --resource-group ${RESOURCE_GROUP} \
    --name ${REGISTRY_NAME} \
    --sku Basic \
    --admin-enabled true

# Build and push image
echo "🔨 Building and pushing image..."
az acr build --registry ${REGISTRY_NAME} \
    --image ${CONTAINER_APP_NAME}:latest \
    --file docker/api.Dockerfile .

# Create container app environment
echo "�� Creating container app environment..."
az containerapp env create --name ${CONTAINER_APP_NAME}-env \
    --resource-group ${RESOURCE_GROUP} \
    --location "${LOCATION}"

# Deploy container app
echo "🚀 Deploying container app..."
az containerapp create --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --environment ${CONTAINER_APP_NAME}-env \
    --image ${REGISTRY_NAME}.azurecr.io/${CONTAINER_APP_NAME}:latest \
    --target-port 8000 \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 10 \
    --cpu 2.0 \
    --memory 4.0Gi

echo "✅ Deployment complete!"
echo "🌐 Service URL: https://${CONTAINER_APP_NAME}.${CONTAINER_APP_NAME}-env.${LOCATION}.azurecontainerapps.io"
