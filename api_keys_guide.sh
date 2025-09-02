#!/bin/bash
# API Key Management Script for Contract Review System

echo "🔑 API Key Management for Contract Review & Risk Analysis System"
echo "================================================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp env_template.txt .env
fi

echo ""
echo "📋 Required API Keys for Next Phases:"
echo ""

echo "1. 🔤 OpenAI API Key (Required for RAG)"
echo "   - Visit: https://platform.openai.com/api-keys"
echo "   - Cost: ~$50-200/month depending on usage"
echo "   - Used for: Alternative clause suggestions, RAG responses"
echo ""

echo "2. ☁️  Azure OpenAI (Optional - Enterprise)"
echo "   - Visit: https://portal.azure.com"
echo "   - Cost: Similar to OpenAI + enterprise features"
echo "   - Used for: Higher rate limits, better security"
echo ""

echo "3. 🗄️  Pinecone API Key (Optional - Production Vector DB)"
echo "   - Visit: https://www.pinecone.io/"
echo "   - Cost: $0.10 per 1K operations (~$20-100/month)"
echo "   - Used for: Similar clause retrieval, precedent database"
echo ""

echo "4. ☁️  AWS/GCP Credentials (Optional - Cloud Storage)"
echo "   - AWS: https://aws.amazon.com/ (S3 for documents)"
echo "   - GCP: https://cloud.google.com/ (Storage for documents)"
echo "   - Cost: ~$5-50/month for storage"
echo "   - Used for: Document storage, model artifacts"
echo ""

echo "5. 📊 Database Credentials (Optional - Production)"
echo "   - PostgreSQL: For metadata storage"
echo "   - Redis: For caching"
echo "   - Cost: $10-100/month depending on size"
echo ""

echo "🚀 Current Phase Status:"
echo "   ✅ Phase 1 (MVP): No external APIs needed"
echo "   🔑 Phase 2 (RAG): OpenAI API key required"
echo "   🔑 Phase 3 (Production): All APIs recommended"
echo ""

echo "💡 Recommendations:"
echo "   - Start with OpenAI API key for Phase 2"
echo "   - Use free tiers where possible"
echo "   - Monitor usage to control costs"
echo "   - Consider Azure for enterprise deployments"
echo ""

echo "🔧 To add API keys:"
echo "   1. Get your API keys from the providers above"
echo "   2. Edit .env file: nano .env"
echo "   3. Replace 'your-xxx-api-key' with actual keys"
echo "   4. Restart the application"
echo ""

echo "📈 Cost Optimization Tips:"
echo "   - Use GPT-3.5 for initial testing (cheaper)"
echo "   - Implement caching to reduce API calls"
echo "   - Batch process contracts when possible"
echo "   - Monitor usage with built-in metrics"
echo ""

echo "🎯 For Employment Portfolio:"
echo "   - Start with free/local development"
echo "   - Add OpenAI API for RAG demo"
echo "   - Show cloud deployment capability"
echo "   - Demonstrate cost awareness"
echo ""

echo "✅ Ready to proceed with current setup!"
echo "   You can start development without external APIs"
echo "   Add APIs as needed for advanced features"
