#!/bin/bash
set -e

echo "🪙 AI Token Cost Explorer — Launch Script"
echo "========================================="

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ python3 not found. Install from https://python.org"
  exit 1
fi

# Check for API key (only warn — tabs 3/4/5 work without it)
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "⚠️  ANTHROPIC_API_KEY is not set."
  echo "   Tabs 1 & 2 (live Claude calls) will not work."
  echo "   Tabs 3, 4, 5 (cost calculators) work fine without it."
  echo ""
  echo "   To set it:  export ANTHROPIC_API_KEY=sk-ant-..."
  echo ""
else
  echo "✅ ANTHROPIC_API_KEY is set"
fi

# Install dependencies if needed
if ! python3 -c "import streamlit, anthropic, pandas, plotly" &>/dev/null; then
  echo ""
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
