# 🚀 Deployment Guide - Streamlit Cloud

## ⚠️ Important: Ollama Limitation

**Problem:** Streamlit Cloud cannot run Ollama because:
- Ollama requires a local server process
- Streamlit Cloud is serverless (no persistent background processes)
- Cannot install and run `ollama serve` on cloud infrastructure

**Solutions:**

### Option 1: **Demo Mode** (Recommended for Submission)
Deploy app with **mock/demo responses** to showcase UI and visualizations.
- ✅ Shows complete UI and features
- ✅ Demonstrates visualizations
- ✅ Perfect for portfolio/demo
- ❌ Not fully functional (no real LLM responses)

### Option 2: **Local + Screen Recording** (Best for Grading)
Run locally and create screen recording.
- ✅ Fully functional
- ✅ Shows real RAG in action
- ✅ Can demonstrate all features
- ❌ No live link

### Option 3: **Use API DIAL** (If you have VPN access)
Replace Ollama with DIAL API in deployment version.
- ✅ Cloud-compatible
- ✅ Fully functional
- ❌ Requires VPN + API DIAL setup
- ❌ More complex configuration

---

## 📋 Submission Strategy

### ✅ **Recommended Approach:**

1. **GitHub Repository** (Required) ✅
   - Push complete code to GitHub
   - Well-documented, annotated source
   - **This is your primary submission**

2. **Local Demo** (Recommended) ✅
   - Run app locally: `./start.sh`
   - Create screen recording showing:
     - RAG query with real Ollama responses
     - RAGAS evaluation results
     - Visualizations
   - **Shows actual functionality**

3. **Streamlit Cloud** (Optional - UI Showcase)
   - Deploy in "demo mode" to show UI
   - Include note: "Requires local Ollama for full functionality"
   - **Shows professional deployment**

---

## 🔧 Option 1: GitHub Deployment (Required)

### Step 1: Prepare Repository

```bash
cd /Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Complete RAG Dog Breed Analyzer with RAGAS evaluation"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create repository: `rag-dog-breed-analyzer`
3. **Do NOT** initialize with README (you already have one)

### Step 3: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer.git

# Push
git branch -M main
git push -u origin main
```

### Step 4: Verify on GitHub

Your repository should show:
- ✅ README.md (with full documentation)
- ✅ src/ folder (all Python modules)
- ✅ requirements.txt
- ✅ All documentation files
- ✅ .gitignore (excluding rag-env/)

**✅ This is your primary submission link!**

---

## 🎥 Option 2: Screen Recording (Recommended)

### What to Record (5-7 minutes):

1. **Introduction** (30 sec)
   - "RAG Dog Breed Analyzer with multimodal capabilities"
   - Show project structure briefly

2. **Initialize RAG** (1 min)
   - Show Ollama running
   - Initialize RAG engine in app
   - Show 64 documents ingested

3. **Query Demo** (2 min)
   - Ask 2-3 questions:
     - "What are good dog breeds for families with children?"
     - "Compare Golden Retriever and Beagle"
     - "Which breeds are hypoallergenic?"
   - Show retrieved contexts
   - Explain reranking

4. **Evaluation** (2 min)
   - Generate evaluation dataset
   - Run RAGAS evaluation (5 samples)
   - Show radar chart
   - Explain metrics

5. **Visualizations** (1 min)
   - Generate embedding space map
   - Show breed clustering
   - Explain UMAP visualization

6. **Code Walkthrough** (1 min)
   - Show key files (rag_engine.py, evaluator.py)
   - Highlight chunking strategies
   - Show multimodal implementation

### Recording Tools:

**macOS:**
```bash
# QuickTime (built-in)
# File → New Screen Recording

# Or OBS Studio (free, professional)
# Download: https://obsproject.com/
```

**Upload to:**
- YouTube (unlisted)
- Vimeo
- Google Drive (with sharing enabled)
- Loom

---

## 🌐 Option 3: Streamlit Cloud (UI Demo)

### A. Create Demo Mode Version

Create a deployment-ready app that works without Ollama:

```bash
cp src/app.py src/app_demo.py
```

Then create a demo mode that shows pre-recorded responses.

### B. Add Deployment Files

Create `streamlit_app.py` (entry point for Streamlit Cloud):

```python
# streamlit_app.py - Cloud deployment entry point
import streamlit as st

st.set_page_config(
    page_title="🐕 RAG Dog Breed Analyzer - Demo",
    page_icon="🐕",
    layout="wide",
)

st.title("🐕 RAG Dog Breed Analyzer")
st.markdown("### Multimodal Document Analysis with Evaluation")

st.warning("""
⚠️ **Demo Mode**: This is a UI demonstration.
For full functionality with Ollama LLM, please run locally.

📥 **Get the full version**: [GitHub Repository](https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer)
""")

st.info("""
**Features demonstrated:**
- ✅ Interactive UI
- ✅ Dataset browsing (64 dog breeds)
- ✅ Visualization examples
- ✅ Code structure
- ⚠️ RAG queries require local Ollama (not available in cloud)
""")

# Import and show dataset info
import json
from pathlib import Path

# Show dataset statistics
if Path("data/breed_mapping.json").exists():
    with open("data/breed_mapping.json") as f:
        data = json.load(f)

    st.success(f"✅ Dataset loaded: {data['total_breeds']} dog breeds")

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "📈 Example Visualizations", "💻 Code"])

    with tab1:
        st.subheader("Dog Breed Dataset")
        import pandas as pd

        breed_df = pd.DataFrame([
            {
                'Breed': b['name'],
                'Images': b['images_count'],
                'Country': b['characteristics'].get('Country of Origin', 'Unknown'),
                'Temperament': b['characteristics'].get('Character Traits', 'Unknown')[:50] + '...',
            }
            for b in data['breeds']
        ])

        st.dataframe(breed_df, use_container_width=True)

    with tab2:
        st.subheader("Example Visualizations")
        st.markdown("""
        When running locally with Ollama:
        - **RAGAS Radar Chart**: Compare RAG strategies
        - **Embedding Space Map**: Visualize breed similarities
        - **Metrics Bar Chart**: Performance comparison
        """)

        # Show example visualization (static image or pre-generated chart)
        st.info("Visualizations are generated in the full local version")

    with tab3:
        st.subheader("Source Code Structure")
        st.code("""
src/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration
├── document_processor.py   # Multimodal processing
├── rag_engine.py          # RAG with ChromaDB + Ollama
├── evaluator.py           # RAGAS evaluation
├── visualizer.py          # Plotly charts
└── eval_dataset_generator.py  # Q&A generation
        """, language="bash")

        st.markdown("""
        **Key Features:**
        - 3 chunking strategies (fixed-size, semantic, combined)
        - Cross-encoder reranking
        - Multimodal RAG (LLaVA vision model)
        - 4 RAGAS metrics (faithfulness, relevancy, precision, recall)
        """)

else:
    st.error("Dataset not found. This demo requires the data/ directory.")

st.divider()

st.markdown("""
### 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer
cd rag-dog-breed-analyzer
source rag-env/bin/activate
pip install -r requirements.txt

# Start Ollama
ollama serve
ollama pull llama3

# Run app
streamlit run src/app.py
```

**Full documentation:** See README.md in repository
""")
```

### C. Deploy to Streamlit Cloud

1. **Push to GitHub** (if not done):
   ```bash
   git add streamlit_app.py
   git commit -m "Add Streamlit Cloud demo mode"
   git push
   ```

2. **Go to Streamlit Cloud:**
   - Visit: https://streamlit.io/cloud
   - Sign in with GitHub

3. **Deploy App:**
   - Click "New app"
   - Repository: `YOUR_USERNAME/rag-dog-breed-analyzer`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Get Your Link:**
   - URL: `https://YOUR_USERNAME-rag-dog-breed-analyzer.streamlit.app`

**Note:** This will show the UI/dataset browser, but won't have functional RAG (since Ollama can't run in cloud).

---

## 📤 What to Submit

### ✅ **Recommended Submission Package:**

1. **GitHub Repository** (REQUIRED)
   ```
   https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer
   ```
   - Complete source code
   - All documentation
   - Data organization scripts
   - README with instructions

2. **Screen Recording** (HIGHLY RECOMMENDED)
   ```
   https://youtu.be/YOUR_VIDEO_ID
   or
   https://vimeo.com/YOUR_VIDEO_ID
   ```
   - 5-7 minute demo
   - Shows actual functionality with Ollama
   - Demonstrates all features

3. **Streamlit Cloud Link** (OPTIONAL)
   ```
   https://YOUR_USERNAME-rag-dog-breed-analyzer.streamlit.app
   ```
   - UI showcase only
   - Dataset browser
   - Note: "Full functionality requires local Ollama"

4. **One-Pager** (OPTIONAL - already done!)
   ```
   Link to PROJECT_SUMMARY.md in your repo
   ```

---

## 📋 Submission Template

Use this in your submission form:

```
RAG Dog Breed Analyzer - Multimodal Document Analysis

GitHub Repository:
https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer

Screen Recording Demo:
https://youtu.be/YOUR_VIDEO_ID
(Shows full functionality with Ollama locally)

Streamlit Cloud Demo (UI only):
https://YOUR_USERNAME-rag-dog-breed-analyzer.streamlit.app
(Note: Requires local Ollama for RAG functionality)

Project Summary:
https://github.com/YOUR_USERNAME/rag-dog-breed-analyzer/blob/main/PROJECT_SUMMARY.md

Key Features:
✅ Multimodal RAG (LLaVA vision model)
✅ 3 chunking strategies + cross-encoder reranking
✅ RAGAS evaluation (4 metrics)
✅ Interactive visualizations (radar chart + embedding map)
✅ 64 dog breeds with 1,002 images

Technology Stack:
- Ollama (Llama3, LLaVA)
- ChromaDB (vector database)
- RAGAS (evaluation framework)
- Streamlit (web interface)
- Sentence Transformers (embeddings)

Installation:
See README.md for complete setup instructions.
Requires local Ollama server for full functionality.
```

---

## 💡 Why This Approach?

### ✅ **Strengths:**

1. **GitHub Repository**
   - Shows complete, well-documented code
   - Reviewers can clone and run locally
   - Demonstrates professional development practices

2. **Screen Recording**
   - Shows ACTUAL functionality (not mock data)
   - Demonstrates real Ollama responses
   - Can explain your work with voiceover

3. **Streamlit Cloud (optional)**
   - Shows you know cloud deployment
   - Professional portfolio piece
   - Easy to share/demo to others

### ⚠️ **Limitations:**

- Streamlit Cloud can't run Ollama (architectural limitation)
- Most AI apps with local models have this issue
- Industry standard is to use cloud APIs for deployment

---

## 🎯 Alternative: Use API DIAL (If Available)

If you have VPN access to EPAM's DIAL:

1. Create `src/config_cloud.py`:
```python
# Cloud configuration using DIAL API
DIAL_API_URL = "https://dial.api.epam.com/v1"
DIAL_API_KEY = "your-key-here"  # Set via Streamlit secrets

# Use DIAL instead of Ollama in cloud
USE_DIAL = True  # Set to True for cloud deployment
```

2. Modify `src/rag_engine.py` to support both:
```python
if config.USE_DIAL:
    # Use DIAL API
    response = requests.post(config.DIAL_API_URL, ...)
else:
    # Use local Ollama
    response = requests.post(config.OLLAMA_BASE_URL, ...)
```

This would allow fully functional cloud deployment!

---

## 🚀 Quick Deploy Checklist

**For GitHub (Required):**
- [ ] Create GitHub account
- [ ] Create new repository
- [ ] Push code: `git push origin main`
- [ ] Verify README displays correctly
- [ ] Copy repository URL

**For Screen Recording (Recommended):**
- [ ] Test app works locally
- [ ] Prepare demo script (what to show)
- [ ] Record 5-7 min video
- [ ] Upload to YouTube/Vimeo
- [ ] Copy video URL

**For Streamlit Cloud (Optional):**
- [ ] Create `streamlit_app.py` demo mode
- [ ] Push to GitHub
- [ ] Deploy on Streamlit Cloud
- [ ] Copy app URL
- [ ] Add note about local requirements

---

## 📚 Next Steps

1. **First: Push to GitHub** (30 minutes)
   ```bash
   git init
   git add .
   git commit -m "Complete RAG application"
   git remote add origin <your-url>
   git push -u origin main
   ```

2. **Then: Create Screen Recording** (1 hour)
   - Run app locally
   - Record demo
   - Upload video

3. **Optional: Deploy demo to Streamlit Cloud** (30 minutes)
   - Create demo mode
   - Deploy to cloud
   - Add limitation notice

**Total time: 2-3 hours for complete submission package**

---

## ❓ Questions?

**Q: Can I deploy to other platforms?**
A: Yes! Alternatives:
- Hugging Face Spaces (same Ollama limitation)
- Render, Railway (can run Docker with Ollama)
- AWS/GCP/Azure (full control, but complex)

**Q: Is GitHub enough for submission?**
A: Yes! GitHub + screen recording is the recommended approach for this type of AI application.

**Q: What if I don't want to create video?**
A: GitHub repository alone is acceptable, but video greatly helps reviewers understand your work.

---

**Ready to deploy?** Start with:

```bash
git init
git add .
git commit -m "Complete RAG Dog Breed Analyzer"
```

Then create your GitHub repository! 🚀
