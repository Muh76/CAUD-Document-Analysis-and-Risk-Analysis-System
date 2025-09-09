#!/bin/bash

# Streamlit Share Deployment Script
set -e

# Configuration
STREAMLIT_APP_FILE="app/ui/app.py"
REQUIREMENTS_FILE="requirements.txt"
SECRETS_FILE=".streamlit/secrets.toml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Preparing Contract Analysis System for Streamlit Share${NC}"

# Check if files exist
if [ ! -f "$STREAMLIT_APP_FILE" ]; then
    echo -e "${RED}❌ Error: Streamlit app file not found: $STREAMLIT_APP_FILE${NC}"
    exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${RED}❌ Error: Requirements file not found: $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Create .streamlit directory
mkdir -p .streamlit

# Create secrets file template
echo -e "${GREEN}📝 Creating secrets template...${NC}"
cat > ${SECRETS_FILE} << EOF
# Streamlit Share Secrets Configuration
# Replace these values with your actual API configuration

[api]
url = "https://your-api-url.com"
token = "your-api-token"

[deployment]
environment = "production"
log_level = "INFO"
EOF

# Create streamlit config
echo -e "${GREEN}⚙️ Creating Streamlit configuration...${NC}"
cat > .streamlit/config.toml << EOF
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
EOF

echo -e "${GREEN}✅ Streamlit Share preparation complete!${NC}"
echo -e "${YELLOW}📋 Next steps:${NC}"
echo -e "${YELLOW}1. Update .streamlit/secrets.toml with your API configuration${NC}"
echo -e "${YELLOW}2. Push your code to GitHub${NC}"
echo -e "${YELLOW}3. Connect your repository to Streamlit Share${NC}"
echo -e "${YELLOW}4. Deploy your app${NC}"
echo -e "${GREEN}🌐 Your app will be available at: https://share.streamlit.io/your-username/your-repo${NC}"
