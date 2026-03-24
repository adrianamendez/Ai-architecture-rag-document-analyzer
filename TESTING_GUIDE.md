# 🧪 Testing Guide - RAG Dog Breed Analyzer

## Prerequisites Checklist

Before testing, ensure you have:

```bash
# 1. Python 3.9+ installed
python3 --version  # Should show 3.9 or higher

# 2. Virtual environment activated
source rag-env/bin/activate
# You should see (rag-env) in your terminal

# 3. Dependencies installed
pip list | grep streamlit  # Should show streamlit

# 4. Ollama installed and running
ollama --version  # Should show version
```

---

## Step-by-Step Testing

### 🟢 Test 1: Environment Setup

```bash
# Navigate to project
cd /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer

# Activate virtual environment
source rag-env/bin/activate

# Verify installation
pip list | grep -E "streamlit|chromadb|ragas|plotly"
```

**Expected Output:**
```
chromadb           0.4.15
plotly             5.17.0
ragas              0.1.0
streamlit          1.28.0
```

**✅ PASS:** All packages listed
**❌ FAIL:** Missing packages → Run `pip install -r requirements.txt`

---

### 🟢 Test 2: Ollama Setup

#### 2a. Start Ollama Server

**Open a SEPARATE terminal window:**
```bash
ollama serve
```

**Expected Output:**
```
Ollama server running on http://localhost:11434
```

**Leave this terminal running!**

#### 2b. Pull Required Models

**In another terminal:**
```bash
# Pull Llama3 (text-only model)
ollama pull llama3

# Pull LLaVA (multimodal model) - OPTIONAL but recommended
ollama pull llava
```

**Expected Output:**
```
pulling manifest
pulling model...
success
```

#### 2c. Verify Models

```bash
ollama list
```

**Expected Output:**
```
NAME            ID              SIZE    MODIFIED
llama3:latest   xxx             4.7 GB  X days ago
llava:latest    xxx             4.5 GB  X days ago
```

**✅ PASS:** llama3 listed (llava is optional)
**❌ FAIL:** No models → Repeat `ollama pull llama3`

---

### 🟢 Test 3: Configuration Module

```bash
cd /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer
python src/config.py
```

**Expected Output:**
```
============================================================
RAG Dog Breed Analyzer - Configuration
============================================================

Project Root: /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer
Data Directory: /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer/data
Vector DB: /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer/data/vector_db

Available Models: ['text_only', 'multimodal', 'multimodal_bakllava']
Chunking Strategies: ['fixed_size', 'semantic', 'combined']
RAGAS Metrics: ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

Ollama Available: True
============================================================
```

**✅ PASS:** Shows "Ollama Available: True"
**❌ FAIL:** Shows "Ollama Available: False" → Check if Ollama server is running

---

### 🟢 Test 4: Document Processor

```bash
python src/document_processor.py
```

**Expected Output:**
```
============================================================
Document Processor Test
============================================================

Testing fixed_size chunking strategy:
------------------------------------------------------------
INFO:__main__:Loaded 64 breeds
INFO:__main__:Processing breed: German Shepherd

German Shepherd: 3 chunks created
  Chunk 1 preview: Breed: German Shepherd
Country of Origin: Germany
Fur Color: Black, Tan...

[... more output ...]

Total documents: 9
Embedding shape: (384,)

✓ Document processing test complete!
============================================================
```

**✅ PASS:** Shows processed documents with embeddings
**❌ FAIL:** Error messages → Check data files exist

---

### 🟢 Test 5: RAG Engine (Most Important!)

```bash
python src/rag_engine.py
```

**Expected Output:**
```
============================================================
RAG Engine Test
============================================================

1. Testing Text-Only RAG (Llama3)
------------------------------------------------------------
INFO:__main__:Initializing ChromaDB at .../data/vector_db
INFO:__main__:Created new collection: dog_breeds_combined
INFO:__main__:Processing all breed documents...
INFO:__main__:Generating embeddings for 64 documents...
Batches: 100%|██████████| 2/2 [00:03<00:00]
INFO:__main__:Ingesting 64 documents into ChromaDB...
INFO:__main__:✓ Ingested 64 documents successfully

Documents in collection: 64

Collection Stats:
  total_documents: 64
  unique_breeds: 64
  collection_name: dog_breeds_combined
  chunking_strategy: combined
  distance_metric: cosine

2. Testing Query
------------------------------------------------------------

Query: What are good dog breeds for families with children?
INFO:__main__:Processing query: What are good dog breeds for families with children?
INFO:__main__:Retrieved 10 documents
INFO:__main__:Reranked 10 documents, returning top 3
INFO:__main__:Generating answer with llama3...

Answer: Based on the provided context, some good dog breeds for families with children include:
1. Beagle - Known to be curious, friendly, energetic, and good-natured
2. Golden Retriever - Intelligent, friendly, kind, loyal, and good-natured
3. Labrador Retriever - (mentioned in context)
...

Metadata:
  Model: llama3
  Retrieved docs: 3
  Reranking used: True
  Context breeds: ['Beagle', 'Golden Retriever', 'Boxer']

✓ RAG Engine test complete!
============================================================
```

**⏱️ This test takes 3-5 minutes** (embedding generation + LLM query)

**✅ PASS:** Successfully ingests docs and generates answer
**❌ FAIL:** Check errors below

**Common Errors:**

1. **"Ollama server not detected"**
   ```bash
   # In separate terminal:
   ollama serve
   ```

2. **"Model not found"**
   ```bash
   ollama pull llama3
   ```

3. **"ChromaDB error"**
   ```bash
   # Delete and retry
   rm -rf data/vector_db/*
   python src/rag_engine.py
   ```

---

### 🟢 Test 6: Evaluation Dataset Generator

```bash
python src/eval_dataset_generator.py
```

**Expected Output:**
```
============================================================
Evaluation Dataset Generator
============================================================

Generating 50 Q&A pairs...
INFO:__main__:Loaded 64 breeds for evaluation dataset
INFO:__main__:Generating 30 specific breed questions...
INFO:__main__:Generating 10 comparison questions...
INFO:__main__:Generating 10 search questions...
INFO:__main__:✓ Generated 50 Q&A pairs

Sample Q&A pairs:
------------------------------------------------------------

1. Type: specific_breed
   Question: What are the characteristics of German Shepherd?
   Answer: German Shepherd is a dog breed from Germany. They have Black, Tan colored fur...

2. Type: comparison
   Question: Compare Beagle and Chihuahua.
   Answer: Beagle and Chihuahua are both popular dog breeds...

3. Type: search
   Question: What dog breeds are from England?
   Answer: Dog breeds from England include: Beagle, Cocker Spaniel...

Saving dataset...
✓ Saved to: .../data/eval_dataset/eval_dataset.json

Dataset Statistics:
------------------------------------------------------------
  specific_breed: 30 questions (60.0%)
  comparison: 10 questions (20.0%)
  search: 10 questions (20.0%)

✓ Evaluation dataset generation complete!
============================================================
```

**✅ PASS:** Creates eval_dataset.json with 50 Q&A pairs
**❌ FAIL:** Check breed_mapping.json exists

---

### 🟢 Test 7: Visualizer

```bash
python src/visualizer.py
```

**Expected Output:**
```
============================================================
Visualizer Test
============================================================

1. Creating sample RAGAS radar chart...
✓ Saved to /tmp/ragas_radar_test.html

2. Creating metrics bar chart...
✓ Saved to /tmp/metrics_bar_test.html

3. Creating comparison table...
                  Strategy  Model  ...  Context Recall  Average
0  Multimodal + Reranking  llava  ...           0.810    0.870
1          Multimodal RAG  llava  ...           0.750    0.820
2           Text-Only RAG llama3  ...           0.680    0.758

4. Creating embedding space map...
✓ Saved to /tmp/embedding_space_test.html

============================================================
✓ Visualizer test complete!
Open the HTML files in your browser to view the charts.
============================================================
```

**✅ PASS:** Creates 3 HTML files in /tmp
**❌ FAIL:** Check plotly installed

**View the charts:**
```bash
# Open in browser
open /tmp/ragas_radar_test.html
open /tmp/metrics_bar_test.html
open /tmp/embedding_space_test.html
```

---

### 🟢 Test 8: Full Streamlit Application

#### 8a. Start the App

```bash
./start.sh
```

OR directly:

```bash
streamlit run src/app.py
```

**Expected Output:**
```
============================================================
🐕 RAG Dog Breed Analyzer - Startup
============================================================

Checking Ollama server...
✓ Ollama server is running

Checking Ollama models...
✓ llama3 found
✓ llava found

Checking data...
✓ Found 64 dog breeds

Checking Python dependencies...
✓ Dependencies installed

============================================================
Starting Streamlit app...
============================================================

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Browser should auto-open to http://localhost:8501**

#### 8b. Test App Interface

**In the Streamlit App:**

1. **Sidebar - Initialize RAG Engine**
   - ✅ Select "text_only" model
   - ✅ Choose "combined" chunking
   - ✅ Enable "Reranking"
   - ✅ Click **"Initialize RAG Engine"**

   **Expected:** Green success message: "✓ Ingested 64 documents"

2. **Query Tab - Test Search**
   - ✅ Type: "What are the characteristics of Golden Retrievers?"
   - ✅ Click **"Search"**

   **Expected:**
   - Answer appears in ~3-5 seconds
   - Shows retrieved contexts
   - Displays metadata (response time, model, docs retrieved)

3. **Evaluation Tab - Generate Dataset**
   - ✅ Click "Generate Dataset" (30 samples)

   **Expected:** "✓ Generated 30 Q&A pairs"

4. **Evaluation Tab - Run Evaluation**
   - ✅ Set samples to "5" (faster test)
   - ✅ Click "Run Evaluation"

   **Expected:**
   - Takes 2-3 minutes
   - Shows radar chart
   - Displays metrics scores

   **⏱️ Note:** This is the slowest test (2-3 min for 5 samples)

5. **Visualizations Tab - Embedding Map**
   - ✅ Select "UMAP" method
   - ✅ Set breeds to "30"
   - ✅ Click "Generate Embedding Map"

   **Expected:**
   - Shows 2D scatter plot
   - Breeds colored by country
   - Interactive plot

6. **Dataset Info Tab**
   - ✅ Browse breeds table
   - ✅ Select a breed to view details

   **Expected:** Shows breed information

---

## 🎯 Quick Test Checklist

Use this for fast verification:

```bash
# 1. Check Ollama (in separate terminal)
ollama serve

# 2. In project directory
cd /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer
source rag-env/bin/activate

# 3. Quick tests (run in order)
python src/config.py           # 2 sec - Check config
python src/document_processor.py  # 30 sec - Test processing
python src/rag_engine.py       # 3-5 min - Full RAG test

# 4. Start app
streamlit run src/app.py       # Interactive testing
```

---

## ✅ Success Criteria

Your system is working if:

- ✅ `config.py` shows "Ollama Available: True"
- ✅ `document_processor.py` generates embeddings
- ✅ `rag_engine.py` ingests 64 documents and answers questions
- ✅ Streamlit app opens at http://localhost:8501
- ✅ Can initialize RAG engine in app
- ✅ Can query and get answers
- ✅ Can generate evaluation dataset
- ✅ Can run evaluation (even with 5 samples)
- ✅ Can generate visualizations

---

## 🐛 Common Issues & Solutions

### Issue 1: "Ollama server not detected"

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Keep this terminal open!
```

### Issue 2: "Model 'llama3' not found"

**Solution:**
```bash
ollama pull llama3
ollama list  # Verify
```

### Issue 3: "ChromaDB permission error"

**Solution:**
```bash
# Delete and recreate
rm -rf data/vector_db/*
# Then reinitialize in app
```

### Issue 4: "Import Error: No module named 'streamlit'"

**Solution:**
```bash
# Activate venv
source rag-env/bin/activate

# Reinstall
pip install -r requirements.txt
```

### Issue 5: "RAGAS evaluation fails"

**Solutions:**
1. Reduce samples to 5 (faster, less likely to fail)
2. Skip evaluation (optional feature)
3. Set dummy OpenAI key if needed:
   ```bash
   export OPENAI_API_KEY="sk-dummy-key"
   ```

### Issue 6: App won't start

**Solution:**
```bash
# Check port 8501 is free
lsof -ti:8501 | xargs kill -9

# Restart
streamlit run src/app.py
```

### Issue 7: "No breeds found"

**Solution:**
```bash
# Check data exists
ls data/images/raw | wc -l  # Should show ~54
cat data/breed_mapping.json | grep "total_breeds"  # Should show 64

# If missing, re-run data organization
python scripts/organize_data.py
```

---

## 📊 Test Results Template

Use this to document your testing:

```
TESTING RESULTS - RAG Dog Breed Analyzer
Date: ___________
Tester: ___________

✅ Test 1: Environment Setup - PASS/FAIL
✅ Test 2: Ollama Setup - PASS/FAIL
✅ Test 3: Configuration - PASS/FAIL
✅ Test 4: Document Processor - PASS/FAIL
✅ Test 5: RAG Engine - PASS/FAIL
✅ Test 6: Eval Dataset Generator - PASS/FAIL
✅ Test 7: Visualizer - PASS/FAIL
✅ Test 8: Streamlit App - PASS/FAIL

Notes:
- Query response time: _____ seconds
- Evaluation time (5 samples): _____ minutes
- Total breeds in system: _____
- RAGAS scores achieved: _____

Overall: PASS / FAIL
```

---

## 🚀 After Testing

Once all tests pass:

1. **Take Screenshots:**
   - App home screen
   - Query results
   - RAGAS radar chart
   - Embedding visualization

2. **Optional: Create Screen Recording**
   ```bash
   # Use QuickTime or OBS to record:
   # 1. Initialize RAG engine
   # 2. Run a query
   # 3. Show evaluation results
   # 4. Display visualizations
   ```

3. **Commit & Push:**
   ```bash
   git add .
   git commit -m "Tested and verified RAG application"
   git push origin main
   ```

4. **Submit!**

---

## 💡 Pro Tips

1. **First time?** Start with quickest tests:
   - `python src/config.py` (2 sec)
   - `streamlit run src/app.py` (30 sec)
   - Test one query (1 min)

2. **Full testing?** Takes about **10-15 minutes total**

3. **In a hurry?** Skip evaluation test (it's the slowest)

4. **Demo for submission?** Focus on:
   - One successful query with answer
   - One RAGAS chart
   - One embedding visualization

5. **Keep Ollama running** in background during testing

---

**Need help?** Check:
- `README.md` for detailed documentation
- `QUICKSTART.md` for setup steps
- Error messages in terminal for specific issues

**Ready to test!** Start with: `./start.sh` 🚀
