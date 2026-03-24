#!/bin/bash

# Startup script for RAG Dog Breed Analyzer
# This script checks prerequisites and starts the Streamlit app

set -e

echo "======================================================================"
echo "🐕 RAG Dog Breed Analyzer - Startup"
echo "======================================================================"

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo ""
    echo "⚠️  Virtual environment not activated!"
    echo "Please run: source rag-env/bin/activate"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Check if Ollama is running
echo ""
echo "Checking Ollama server..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama server is running"
else
    echo "✗ Ollama server not detected"
    echo ""
    echo "Please start Ollama in another terminal:"
    echo "  ollama serve"
    echo ""
    read -p "Press Enter when Ollama is running, or Ctrl+C to exit..."
fi

# Check if required models are available
echo ""
echo "Checking Ollama models..."
MODELS=$(ollama list 2>/dev/null || echo "")

if echo "$MODELS" | grep -q "llama3"; then
    echo "✓ llama3 found"
else
    echo "⚠️  llama3 not found"
    echo "Run: ollama pull llama3"
fi

if echo "$MODELS" | grep -q "llava"; then
    echo "✓ llava found"
else
    echo "⚠️  llava not found (optional for multimodal)"
    echo "Run: ollama pull llava"
fi

# Check if data exists
echo ""
echo "Checking data..."
if [ -f "data/breed_mapping.json" ]; then
    BREED_COUNT=$(python3 -c "import json; data=json.load(open('data/breed_mapping.json')); print(data['total_breeds'])")
    echo "✓ Found $BREED_COUNT dog breeds"
else
    echo "✗ Data not organized"
    echo "Run: python scripts/organize_data.py"
    exit 1
fi

# Check Python dependencies
echo ""
echo "Checking Python dependencies..."
if python3 -c "import streamlit, chromadb, sentence_transformers, plotly" 2>/dev/null; then
    echo "✓ Dependencies installed"
else
    echo "⚠️  Some dependencies missing"
    echo "Run: pip install -r requirements.txt"
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

# Start Streamlit
echo ""
echo "======================================================================"
echo "Starting Streamlit app..."
echo "======================================================================"
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "To stop the app, press Ctrl+C in this terminal"
echo ""

streamlit run src/app.py
