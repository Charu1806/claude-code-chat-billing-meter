#!/bin/bash
set -e

echo "🪙 AI Token Cost Explorer — Launch Script"
echo "========================================="

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Install from https://python.org"
  exit 1
fi

# Create .env from example if it doesn't exist yet
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "📄 Created .env from .env.example — edit it to set your API key"
  fi
else
  echo "✅ .env file found"
fi

# Load .env to check the key in this shell too
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true
fi

# Check for API key (only warn — tabs 3/4/5 work without it)
if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "sk-ant-..." ]; then
  echo ""
  echo "⚠️  ANTHROPIC_API_KEY is not set in .env"
  echo "   Tabs 1 & 2 (live Claude calls) will not work."
  echo "   Tabs 3, 4, 5 (cost calculators) work fine without it."
  echo ""
  echo "   To fix: open .env and replace sk-ant-... with your real key"
  echo "   Get a key → https://console.anthropic.com/settings/keys"
  echo ""
else
  echo "✅ ANTHROPIC_API_KEY is set"
fi

# Show which models are configured
echo ""
echo "📋 Model config (from .env):"
echo "   Code Writer  → ${CODE_WRITER_MODEL:-claude-sonnet-4-6 (default)}"
echo "   Chat default → ${CHAT_DEFAULT_MODEL:-claude-sonnet-4-6 (default)}"
echo ""

# Install dependencies if needed
if ! python3 -c "import streamlit, anthropic, pandas, plotly, dotenv" &>/dev/null; then
  echo "📦 Installing dependencies..."
  pip3 install -r requirements.txt
else
  echo "✅ Dependencies already installed"
fi

echo ""
echo "🚀 Starting app at http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run app.py
