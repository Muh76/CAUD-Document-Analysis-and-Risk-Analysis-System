#!/bin/bash
# Setup API Keys for Contract Review System

echo "🔑 Setting up API Keys for Contract Review & Risk Analysis System"
echo "================================================================"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp env_template.txt .env
    echo "✅ Created .env file from template"
fi

# Update OpenAI API key
echo "🔤 Setting up OpenAI API key..."
sed -i '' 's/OPENAI_API_KEY=your-openai-api-key-here/OPENAI_API_KEY=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/' .env

echo "✅ OpenAI API key configured!"
echo ""

echo "📋 Next Steps for Azure & Google Cloud:"
echo ""

echo "🔵 Azure Setup:"
echo "   1. Go to: https://portal.azure.com"
echo "   2. Create OpenAI resource"
echo "   3. Get endpoint URL and API key"
echo "   4. Update .env file with:"
echo "      AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/"
echo "      AZURE_OPENAI_API_KEY=your-azure-key"
echo ""

echo "🟢 Google Cloud Setup:"
echo "   1. Go to: https://console.cloud.google.com"
echo "   2. Create project and enable APIs"
echo "   3. Create service account and download JSON"
echo "   4. Update .env file with:"
echo "      GOOGLE_CLOUD_PROJECT=your-project-id"
echo "      GOOGLE_CLOUD_STORAGE_BUCKET=your-bucket-name"
echo "      GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json"
echo ""

echo "💡 Optional Services (Free Tiers Available):"
echo ""

echo "🗄️  Vector Database Options:"
echo "   - ChromaDB: FREE (local/self-hosted)"
echo "   - Pinecone: FREE tier (1 index, 100K vectors)"
echo "   - Weaviate: FREE tier (cloud)"
echo ""

echo "☁️  Cloud Storage Options:"
echo "   - AWS S3: FREE tier (5GB for 12 months)"
echo "   - Google Cloud Storage: FREE tier (5GB for 12 months)"
echo "   - Azure Blob Storage: FREE tier (5GB for 12 months)"
echo ""

echo "📊 Database Options:"
echo "   - PostgreSQL: FREE (local) or $10-50/month (cloud)"
echo "   - Redis: FREE (local) or $15-100/month (cloud)"
echo ""

echo "🎯 For Employment Portfolio:"
echo "   ✅ OpenAI API: CONFIGURED"
echo "   🔵 Azure: RECOMMENDED (enterprise focus)"
echo "   🟢 Google Cloud: RECOMMENDED (tech company focus)"
echo "   🗄️  Vector DB: OPTIONAL (ChromaDB free)"
echo "   ☁️  Cloud Storage: OPTIONAL (free tiers available)"
echo ""

echo "🚀 Ready to start development!"
echo "   Run: ./start.sh to launch the application"
