#!/bin/bash

# Deployment Validation Script
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🔍 Validating Contract Analysis System Deployment${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required tools
echo -e "${YELLOW}📋 Checking required tools...${NC}"

if command_exists docker; then
    echo -e "${GREEN}✅ Docker is installed${NC}"
else
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if command_exists gcloud; then
    echo -e "${GREEN}✅ Google Cloud CLI is installed${NC}"
else
    echo -e "${YELLOW}⚠️ Google Cloud CLI is not installed (required for GCP deployment)${NC}"
fi

if command_exists az; then
    echo -e "${GREEN}✅ Azure CLI is installed${NC}"
else
    echo -e "${YELLOW}⚠️ Azure CLI is not installed (required for Azure deployment)${NC}"
fi

# Check configuration files
echo -e "${YELLOW}📋 Checking configuration files...${NC}"

if [ -f "pyproject.toml" ]; then
    echo -e "${GREEN}✅ pyproject.toml exists${NC}"
else
    echo -e "${RED}❌ pyproject.toml not found${NC}"
    exit 1
fi

if [ -f "docker/api.Dockerfile" ]; then
    echo -e "${GREEN}✅ API Dockerfile exists${NC}"
else
    echo -e "${RED}❌ API Dockerfile not found${NC}"
    exit 1
fi

if [ -f "docker/ui.Dockerfile" ]; then
    echo -e "${GREEN}✅ UI Dockerfile exists${NC}"
else
    echo -e "${RED}❌ UI Dockerfile not found${NC}"
    exit 1
fi

if [ -f "docker-compose.yml" ]; then
    echo -e "${GREEN}✅ Docker Compose file exists${NC}"
else
    echo -e "${RED}❌ Docker Compose file not found${NC}"
    exit 1
fi

# Check app structure
echo -e "${YELLOW}📋 Checking app structure...${NC}"

if [ -d "app" ]; then
    echo -e "${GREEN}✅ App directory exists${NC}"
else
    echo -e "${RED}❌ App directory not found${NC}"
    exit 1
fi

if [ -f "app/__init__.py" ]; then
    echo -e "${GREEN}✅ App package initialized${NC}"
else
    echo -e "${RED}❌ App package not initialized${NC}"
    exit 1
fi

# Test Docker build
echo -e "${YELLOW}📋 Testing Docker build...${NC}"
if docker build -f docker/api.Dockerfile -t contract-analysis-test . > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker build successful${NC}"
    docker rmi contract-analysis-test > /dev/null 2>&1
else
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All validation checks passed!${NC}"
echo -e "${GREEN}🚀 System is ready for deployment${NC}"
