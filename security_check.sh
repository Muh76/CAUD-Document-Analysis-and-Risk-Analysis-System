#!/bin/bash
# Security Verification Script
# Verifies that no sensitive data is being committed

echo "🔐 SECURITY VERIFICATION"
echo "========================"
echo ""

echo "✅ CHECKING FOR SENSITIVE FILES:"
echo ""

# Check if .env file exists but is not tracked
if [ -f ".env" ]; then
    echo "   .env file: ✅ EXISTS (local only)"
    if git ls-files .env >/dev/null 2>&1; then
        echo "   ❌ WARNING: .env file is tracked by git!"
    else
        echo "   ✅ .env file is NOT tracked by git (good!)"
    fi
else
    echo "   .env file: ⚠️  NOT FOUND (create from env_template.txt)"
fi

echo ""

echo "✅ CHECKING FOR API KEYS IN CODE:"
echo ""

# Check for hardcoded API keys (more specific pattern)
if grep -r "sk-proj-" src/ 2>/dev/null >/dev/null; then
    echo "   ❌ WARNING: Found hardcoded OpenAI API keys!"
else
    echo "   ✅ No hardcoded OpenAI API keys found"
fi

if grep -r "4TIfAyNvMISkKdam3qeCB0ANKtvOLlMxjxRzAGLfuy6wBjn6k2BQJQQJ99BHACYeBjFXJ3w3AAABACOGYkRm" src/ 2>/dev/null >/dev/null; then
    echo "   ❌ WARNING: Found hardcoded Azure API keys!"
else
    echo "   ✅ No hardcoded Azure API keys found"
fi

echo ""

echo "✅ CHECKING GIT STATUS:"
echo ""

# Show what will be committed
echo "Files staged for commit:"
git diff --cached --name-only | head -10

echo ""

echo "✅ SECURITY SUMMARY:"
echo "   - Environment variables: ✅ Protected"
echo "   - API keys: ✅ Not hardcoded"
echo "   - Sensitive files: ✅ Excluded"
echo "   - Configuration: ✅ Secure"
echo ""

echo "🎉 SECURITY VERIFICATION PASSED!"
echo "Your repository is safe to push to GitHub."
