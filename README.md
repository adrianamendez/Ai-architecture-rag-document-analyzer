<<<<<<< HEAD
# Denise_mendez_ai_architect



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.epam.com/Denise_Mendez/denise_mendez_ai_architect.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://git.epam.com/Denise_Mendez/denise_mendez_ai_architect/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
=======
# 🐕 RAG Dog Breed Analyzer

A multimodal Retrieval-Augmented Generation (RAG) system for analyzing dog breed information with comprehensive evaluation using RAGAS metrics.

## 📋 Overview

This application demonstrates advanced RAG techniques including:
- **Multimodal RAG**: Text + Image understanding using LLaVA/Llama3 (Ollama)
- **Multiple Chunking Strategies**: Fixed-size, semantic, and combined chunking
- **Reranking**: Cross-encoder reranking for improved retrieval precision
- **RAGAS Evaluation**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- **Interactive Visualizations**: Radar charts and embedding space maps

## 🎯 Features

### ✅ Core Functionality
- **64 Dog Breeds** with text characteristics and images
- **Vector Database** (ChromaDB) with efficient retrieval
- **Cross-Encoder Reranking** for improved accuracy
- **Ollama Integration** for local LLM inference (no API keys required!)

### 📊 Evaluation Metrics (RAGAS)
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Were the right docs retrieved?
- **Context Recall**: Were all relevant docs found?

### 🎨 Visualizations
- **RAGAS Radar Chart**: Compare different RAG strategies
- **Embedding Space Map**: Visualize breed similarities (UMAP/t-SNE)
- **Metrics Comparison**: Bar charts and tables

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama from https://ollama.ai

   # Pull required models
   ollama pull llama3
   ollama pull llava

   # Start Ollama server
   ollama serve
   ```

### Installation

```bash
# 1. Clone/navigate to project directory
cd rag-document-analyzer

# 2. Create virtual environment (if not exists)
python3 -m venv rag-env
source rag-env/bin/activate  # On Windows: rag-env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify data is organized
ls data/images/raw  # Should show breed folders
ls data/documents   # Should show dog_breeds.csv
```

### Running the Application

```bash
# Start Streamlit app
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
rag-document-analyzer/
├── src/
│   ├── app.py                      # Streamlit web application
│   ├── config.py                   # Configuration and settings
│   ├── document_processor.py       # Multimodal document ingestion
│   ├── rag_engine.py              # RAG with ChromaDB + reranking
│   ├── evaluator.py               # RAGAS evaluation metrics
│   ├── visualizer.py              # Plotly visualizations
│   └── eval_dataset_generator.py  # Generate Q&A pairs
│
├── data/
│   ├── images/raw/                # Dog breed images (64 breeds)
│   ├── documents/                 # Breed text data (CSV)
│   ├── breed_mapping.json         # Links images + text
│   ├── vector_db/                 # ChromaDB storage (auto-generated)
│   └── eval_dataset/              # Evaluation Q&A pairs (auto-generated)
│
├── scripts/
│   └── organize_data.py           # Data organization script
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🎮 Usage Guide

### 1. Initialize RAG Engine

In the Streamlit sidebar:
1. Select **Model Type** (text_only, multimodal, etc.)
2. Choose **Chunking Strategy** (fixed_size, semantic, combined)
3. Enable/disable **Reranking**
4. Click **"Initialize RAG Engine"**

### 2. Query Dog Breeds

In the **Query** tab:
- Type your question or click sample questions
- View the generated answer
- Inspect retrieved contexts and metadata
- See response time and images used

**Sample Questions:**
- "What are the characteristics of Golden Retrievers?"
- "Which dog breeds are hypoallergenic?"
- "Compare Beagle and Chihuahua"
- "What breeds are good with children?"

### 3. Run Evaluation

In the **Evaluation** tab:
1. **Generate Evaluation Dataset** (30-50 Q&A pairs)
2. **Run Evaluation** on subset (10-20 samples recommended)
3. View RAGAS metrics in radar chart
4. Compare different RAG strategies

### 4. Visualize Embeddings

In the **Visualizations** tab:
- Generate embedding space maps (UMAP or t-SNE)
- See how breeds cluster by similarity
- Adjust number of breeds to visualize

### 5. Browse Dataset

In the **Dataset Info** tab:
- View all 64 breeds
- Explore breed characteristics
- See dataset statistics

## 🧪 Testing Individual Components

```bash
# Test configuration
python src/config.py

# Test document processor
python src/document_processor.py

# Test RAG engine
python src/rag_engine.py

# Generate evaluation dataset
python src/eval_dataset_generator.py

# Test evaluator
python src/evaluator.py

# Test visualizations
python src/visualizer.py
```

## 📊 Dataset Information

- **Source**: Kaggle Dog Breeds Image Dataset + Custom CSV
- **Total Breeds**: 64 (with both images and text)
- **Total Images**: ~1,000 images
- **Characteristics**: Country, fur color, height, temperament, health issues

## 🔧 Configuration

Edit `src/config.py` to customize:
- Embedding models
- Chunking parameters
- Retrieval settings
- Reranking configuration
- Visualization options
- Ollama endpoints

## 📈 Evaluation Results

The system compares different RAG strategies:

| Strategy | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|----------|-------------|------------------|-------------------|----------------|
| Text-only | 0.85 | 0.78 | 0.72 | 0.68 |
| Multimodal | 0.88 | 0.82 | 0.79 | 0.75 |
| Multimodal + Rerank | 0.92 | 0.89 | 0.86 | 0.81 |

*(Example scores - run your own evaluation for actual results)*

## 🛠️ Advanced Usage

### Custom Evaluation Dataset

Edit `src/eval_dataset_generator.py` to:
- Add custom question templates
- Modify Q&A generation logic
- Include domain-specific queries

### Add More Breeds

1. Add breed images to `data/images/raw/breed_name/`
2. Update `data/documents/dog_breeds.csv`
3. Run `python scripts/organize_data.py`
4. Reinitialize RAG engine in the app

### Change LLM Models

In `src/config.py`, add new models to `MODELS` dict:
```python
"custom_model": {
    "name": "mistral",
    "description": "Custom model",
    "supports_vision": False,
    "temperature": 0.7,
}
```

## 📝 Requirements

All task requirements addressed:
- ✅ **GenAI Framework**: Ollama (local, free, no API keys)
- ✅ **Document Analysis**: 64 dog breeds with text + images
- ✅ **Data Extraction**: Breed characteristics, temperament, health info
- ✅ **Visualization**: RAGAS radar charts + embedding space maps
- ✅ **Evaluation Metrics**: Faithfulness, relevancy, precision, recall
- ✅ **Evaluation Dataset**: 50+ Q&A pairs with ground truth

### Ninja Challenges:
- ⚠️ **Corpus Update**: Can add breeds without full rebuild (incremental ingestion)
- ❌ **Access Control**: Not implemented (could add user-based filtering)
- ✅ **Multi-modal RAG**: Images + text with LLaVA vision model
- ✅ **RAG Evaluation**: Full RAGAS metrics suite

## 🐛 Troubleshooting

### Ollama not available
```bash
# Start Ollama
ollama serve

# Verify models are pulled
ollama list

# Pull if missing
ollama pull llama3
ollama pull llava
```

### ChromaDB errors
```bash
# Delete and rebuild vector DB
rm -rf data/vector_db/*
# Then reinitialize in the app
```

### RAGAS evaluation fails
- Requires OpenAI API key for some metrics (can set dummy key)
- Or use simplified metrics in `src/evaluator.py`

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📚 Resources

- **RAGAS Documentation**: https://docs.ragas.io/
- **Ollama**: https://ollama.ai/
- **ChromaDB**: https://www.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/darshanthakare/dog-breeds-image-dataset

## 🤝 Contributing

This is a training project. Improvements welcome:
- Add more breeds
- Improve evaluation metrics
- Add new visualizations
- Optimize retrieval strategies

## 📄 License

Educational use only. Dog breed data from Kaggle (see attribution).

## 👥 Author

Created for AI/RAG training course.

---

**Questions?** Check the inline documentation in each module!
>>>>>>> 24b28f0 (add HW 2 RAG - first iteration)
