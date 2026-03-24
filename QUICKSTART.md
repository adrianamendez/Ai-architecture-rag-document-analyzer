# 🚀 Quick Start Guide

## Step 1: Install Ollama

```bash
# Download from https://ollama.ai or:
curl https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama3
ollama pull llava
```

## Step 2: Start Ollama Server

**In a separate terminal:**
```bash
ollama serve
```

Leave this running while using the app.

## Step 3: Activate Virtual Environment

```bash
cd rag-document-analyzer
source rag-env/bin/activate
```

## Step 4: Install Dependencies (if not done)

```bash
pip install -r requirements.txt
```

## Step 5: Run the App

**Option A - Use startup script:**
```bash
./start.sh
```

**Option B - Direct Streamlit:**
```bash
streamlit run src/app.py
```

## Step 6: Use the Application

The app will open at `http://localhost:8501`

### First Time Setup:

1. **In Sidebar:**
   - Select "text_only" model (faster for testing)
   - Choose "combined" chunking
   - Enable "Reranking"
   - Click **"Initialize RAG Engine"**
   - Wait for documents to ingest (~30 seconds)

2. **Query Tab:**
   - Try a sample question: "What are good dog breeds for families?"
   - View the answer and retrieved contexts

3. **Evaluation Tab:**
   - Click "Generate Dataset" (30 samples)
   - Click "Run Evaluation" (10 samples)
   - View RAGAS radar chart

4. **Visualizations Tab:**
   - Click "Generate Embedding Map"
   - Explore breed clusters

## Common Commands

```bash
# Activate environment
source rag-env/bin/activate

# Start app
streamlit run src/app.py

# Run tests
python src/config.py
python src/document_processor.py
python src/rag_engine.py

# Generate evaluation data
python src/eval_dataset_generator.py

# Check Ollama models
ollama list

# Pull new model
ollama pull mistral
```

## Troubleshooting

### "Ollama server not detected"
```bash
# In another terminal:
ollama serve
```

### "ChromaDB error"
```bash
# Delete and rebuild
rm -rf data/vector_db/*
# Then reinitialize in app
```

### "Import errors"
```bash
pip install -r requirements.txt --upgrade
```

### "RAGAS evaluation fails"
- Set dummy OpenAI key or skip certain metrics
- Reduce evaluation samples to 5-10

## What to Submit

For your assignment:

1. **GitHub/GitLab Repository** with:
   - All source code (`src/` folder)
   - README.md
   - requirements.txt
   - Sample evaluation results

2. **Streamlit App** (optional):
   - Deploy to Streamlit Cloud
   - Share link

3. **Screen Recording** (optional):
   - Demo query functionality
   - Show evaluation metrics
   - Explain visualizations

4. **One-Pager** (optional):
   - Architecture diagram
   - Key metrics
   - Results summary

## Key Features to Demonstrate

✅ **Multimodal RAG**: Show image + text retrieval
✅ **Chunking Strategies**: Compare 3 different strategies
✅ **Reranking**: Show with/without reranking comparison
✅ **RAGAS Metrics**: Display radar chart
✅ **Embedding Visualization**: Show breed clustering

## Tips

- Start with **text_only** model (faster)
- Use **10 samples** for evaluation (quicker)
- Test with **simple questions** first
- Check Ollama logs if generation fails
- Use **combined chunking** for best results

---

**Need Help?** Check README.md for detailed documentation!
