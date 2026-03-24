#!/bin/bash

# Quick Test Script for RAG Dog Breed Analyzer
# Tests all core components automatically

set -e

PASSED=0
FAILED=0

echo "======================================================================"
echo "🧪 RAG Dog Breed Analyzer - Quick Test Suite"
echo "======================================================================"
echo ""

# Function to print test results
pass() {
    echo "✅ PASS: $1"
    ((PASSED++))
}

fail() {
    echo "❌ FAIL: $1"
    ((FAILED++))
}

# Test 1: Check Python version
echo "Test 1: Python Version"
echo "----------------------------------------------------------------------"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if (( $(echo "$PYTHON_VERSION >= 3.9" | bc -l) )); then
    pass "Python version $PYTHON_VERSION is compatible"
else
    fail "Python version $PYTHON_VERSION is too old (need 3.9+)"
fi
echo ""

# Test 2: Check virtual environment
echo "Test 2: Virtual Environment"
echo "----------------------------------------------------------------------"
if [[ -n "$VIRTUAL_ENV" ]]; then
    pass "Virtual environment is activated: $VIRTUAL_ENV"
else
    fail "Virtual environment not activated. Run: source rag-env/bin/activate"
fi
echo ""

# Test 3: Check Ollama server
echo "Test 3: Ollama Server"
echo "----------------------------------------------------------------------"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    pass "Ollama server is running on port 11434"
else
    fail "Ollama server not detected. Start with: ollama serve"
fi
echo ""

# Test 4: Check Ollama models
echo "Test 4: Ollama Models"
echo "----------------------------------------------------------------------"
MODELS=$(ollama list 2>/dev/null || echo "")
if echo "$MODELS" | grep -q "llama3"; then
    pass "llama3 model found"
else
    fail "llama3 model not found. Run: ollama pull llama3"
fi

if echo "$MODELS" | grep -q "llava"; then
    pass "llava model found (optional but recommended)"
else
    echo "⚠️  WARNING: llava model not found (optional). Run: ollama pull llava"
fi
echo ""

# Test 5: Check dependencies
echo "Test 5: Python Dependencies"
echo "----------------------------------------------------------------------"
DEPS=("streamlit" "chromadb" "sentence-transformers" "plotly" "ragas")
ALL_DEPS_OK=true

for dep in "${DEPS[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        echo "  ✓ $dep installed"
    else
        echo "  ✗ $dep missing"
        ALL_DEPS_OK=false
    fi
done

if $ALL_DEPS_OK; then
    pass "All required dependencies installed"
else
    fail "Some dependencies missing. Run: pip install -r requirements.txt"
fi
echo ""

# Test 6: Check data files
echo "Test 6: Data Files"
echo "----------------------------------------------------------------------"
if [ -f "data/breed_mapping.json" ]; then
    BREED_COUNT=$(python3 -c "import json; data=json.load(open('data/breed_mapping.json')); print(data['total_breeds'])" 2>/dev/null || echo "0")
    if [ "$BREED_COUNT" -gt 0 ]; then
        pass "Found $BREED_COUNT dog breeds in dataset"
    else
        fail "breed_mapping.json exists but has no breeds"
    fi
else
    fail "data/breed_mapping.json not found. Run: python scripts/organize_data.py"
fi

if [ -f "data/documents/dog_breeds.csv" ]; then
    pass "dog_breeds.csv found"
else
    fail "dog_breeds.csv not found"
fi

IMAGE_COUNT=$(find data/images/raw -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
if [ "$IMAGE_COUNT" -gt 0 ]; then
    pass "Found $IMAGE_COUNT dog images"
else
    fail "No images found in data/images/raw"
fi
echo ""

# Test 7: Test configuration module
echo "Test 7: Configuration Module"
echo "----------------------------------------------------------------------"
if python3 src/config.py > /tmp/config_test.txt 2>&1; then
    if grep -q "Ollama Available: True" /tmp/config_test.txt; then
        pass "Configuration module works and Ollama is available"
    else
        fail "Configuration module works but Ollama not available"
    fi
else
    fail "Configuration module failed"
fi
echo ""

# Test 8: Test document processor (quick test)
echo "Test 8: Document Processor (this may take 30 seconds)"
echo "----------------------------------------------------------------------"
if timeout 60 python3 -c "
from src.document_processor import DocumentProcessor
processor = DocumentProcessor(chunking_strategy='combined')
breeds = list(processor.breed_data.items())[:2]
docs = []
for name, info in breeds:
    docs.extend(processor.process_breed(name, info))
print(f'Created {len(docs)} document chunks')
" > /tmp/processor_test.txt 2>&1; then
    CHUNK_COUNT=$(cat /tmp/processor_test.txt | grep "Created" | grep -oE '[0-9]+')
    if [ -n "$CHUNK_COUNT" ] && [ "$CHUNK_COUNT" -gt 0 ]; then
        pass "Document processor created $CHUNK_COUNT chunks"
    else
        fail "Document processor didn't create chunks"
    fi
else
    fail "Document processor test failed"
fi
echo ""

# Summary
echo "======================================================================"
echo "📊 Test Summary"
echo "======================================================================"
echo "Tests Passed: $PASSED"
echo "Tests Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests passed! Your system is ready."
    echo ""
    echo "Next steps:"
    echo "  1. Run the app: ./start.sh"
    echo "  2. Or run directly: streamlit run src/app.py"
    echo ""
    exit 0
else
    echo "⚠️  Some tests failed. Please fix the issues above."
    echo ""
    echo "Common fixes:"
    echo "  - Activate venv: source rag-env/bin/activate"
    echo "  - Install deps: pip install -r requirements.txt"
    echo "  - Start Ollama: ollama serve (in separate terminal)"
    echo "  - Pull models: ollama pull llama3"
    echo ""
    exit 1
fi
