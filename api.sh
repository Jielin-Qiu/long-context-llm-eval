#!/bin/bash
# AgentCodeEval API Keys Setup Script
# 🏆 Sets up all API keys for our 3 Elite Models

echo "🚀 Setting up AgentCodeEval API Keys..."
echo "🏆 Configuring 3 Elite Models:"
echo "   ✅ OpenAI o3 (Direct API)"
echo "   ✅ Claude Sonnet 4 (AWS Bedrock)"  
echo "   ✅ Gemini 2.5 Pro (Direct API)"
echo

# =============================================================================
# 🏆 ELITE MODEL 1: OpenAI o3 (Direct API)
# =============================================================================
export OPENAI_API_KEY="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"

# =============================================================================
# 🏆 ELITE MODEL 2: Claude Sonnet 4 (AWS Bedrock)
# =============================================================================
export AWS_ACCESS_KEY_ID="ASIATJMTYALMIGVLA5KJ"
export AWS_SECRET_ACCESS_KEY="+s/vIDr6/GARw/zLfK+KUdJkNLBLG7VpSbrQYYL1"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjELb//////////wEaCXVzLWVhc3QtMSJHMEUCIQCEJzc3tpoEjY6+z4DL4dhJuPFUT8A+tHAoJAdFTC27JQIgNZeihBiSSgOVg5B9LAL+KS1ycWlHb6jQotLWrjkyHWcqlAMI3///////////ARAEGgwyMjYzMzI1MDg4ODgiDLk0U+fazfU+JTO9SyroAiCPJFld3Ar+Ihl2NZNUekG9+pbKjirEohEjFOaDg5L8vTGYqAY06JmTtoVCPSK8GV2LH9dsQgePaEqh6h4qOHNnGA6TBU1sNUg7gIWE3GViiUFQt89TA5N6jwBvN36ECHm6l22XxIdZlfcNmBLnkkHW45P+0rPKz6c0eofeOtPWHMFqVaDM4+kJDq3UmrdPVDN5FtTleKzQgIUlcilOZned4LhrKlG/FehMSJkYDFfxVesGv3wI2uuq1SKUyxUFpHvO7CbVtnifTP67ysjp3UX+CSnL6vN6rJuYxKrU21d8sDguNAjzN2IUFJABTVp2EMZrb1p+oliYrK/mPKxNWUjLNFFJ8GB7g/9qL9ZHXVZcKh1BonD1Ix26DbewjlKt31kcv0TBqIwO/BABQ+iLkmKCeRTgc/uY/jqkvr2zUwRFXz0afcwFbzrypuN6me7I55naIMKLRkjiWzCCpBMDIZZeMo6SEqq1bzCJy6/EBjreAbSVzVlT+GefJ8h8mJ4tblfdugkalkbwrWw3siWiZNnNldG/+oKl1WSpaCBMCVgCdlYrYOjTpymvnTu6webr3Z+GrcbIYAR14xI1dASoUNmZI2M4LI1WfCeIl7nc/UDPFMBtqtvHZyArXU40lZA/NFVsFgcciZstnQ6q8pwaj2BFqL3YN6bWL+FUehFua22DHl1wKSnGKOj0SLkdri4lzI2STcWOPq2bLMjznFT+EP+4OK3wpnqrbEqV4sdSXuxTKJivPDMqIyCD+oGvVYCLW2Hry5/w8N/EWle3OQABCw=="

# =============================================================================
# 🏆 ELITE MODEL 3: Gemini 2.5 Pro (Direct API)  
# =============================================================================
export GEMINI_API_KEY="AIzaSyA-_6BL5IOea4uESsN1G2ahbRTV1-MQj-g"

# =============================================================================
# 🎯 Verification & Status
# =============================================================================
echo "✅ API Keys configured successfully!"
echo
echo "🔧 Verification:"
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:15}..."
echo "   AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID:0:15}..."
echo "   AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY:0:15}..."
echo "   AWS_SESSION_TOKEN: ${AWS_SESSION_TOKEN:0:30}..."
echo "   GEMINI_API_KEY: ${GEMINI_API_KEY:0:15}..."
echo
echo "🚀 Ready to use AgentCodeEval with 3 Elite Models!"
echo "   Run: agentcodeeval status"
echo "   Run: python test_claude_approaches.py"
echo "   Run: agentcodeeval generate --phase 1"
echo 