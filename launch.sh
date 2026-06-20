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
    echo "📄 Created .env from .env.example — edit it to add your API keys"
  fi
else
  echo "✅ .env file found"
fi

# Load .env into this shell
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -v '^$' | xargs) 2>/dev/null || true
fi

# Check each provider key
echo ""
echo "🔑 API Key Status:"

check_key() {
  local name=$1 val=$2 url=$3
  if [ -z "$val" ] || [[ "$val" == *"..." ]]; then
    echo "   ❌ $name — not set  ($url)"
  else
    echo "   ✅ $name — set"
  fi
}

check_key "ANTHROPIC_API_KEY (Claude Opus/Sonnet/Haiku)" "$ANTHROPIC_API_KEY" "https://console.anthropic.com/settings/keys"
check_key "GOOGLE_API_KEY    (Gemini 2.5 Flash/Pro)    " "$GOOGLE_API_KEY"    "https://aistudio.google.com/app/apikey"
check_key "GROQ_API_KEY      (Mistral, Llama via Groq) " "$GROQ_API_KEY"      "https://console.groq.com/keys"

echo ""
echo "   ℹ️  Tabs 3, 4, 5 (cost calculators) work without any keys."
echo ""

# Show default model
echo "📋 Default active model: ${CODE_WRITER_MODEL:-claude-sonnet-4-6}"
echo ""

# Install dependencies if needed
if ! python3 -c "import streamlit, anthropic, openai, pandas, plotly, dotenv" &>/dev/null; then
  echo "📦 Installing dependencies..."
  pip3 install -r requirements.txt
else
  echo "✅ Dependencies already installed"
fi

echo ""
echo "🚀 Starting app at http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

python3 -m streamlit run app.py
