"""
Streamlit Application for RAG Dog Breed Analyzer.
Interactive interface for querying, evaluating, and visualizing multimodal RAG.
"""

import streamlit as st
import json
import pandas as pd
import time

from config import (
    UI_CONFIG,
    MODELS,
    CHUNKING_STRATEGIES,
    EVAL_DIR,
    BREED_MAPPING_FILE,
    check_ollama_available,
)
from rag_engine import RAGEngine
from evaluator import RAGEvaluator
from visualizer import RAGVisualizer
from eval_dataset_generator import EvalDatasetGenerator

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG['title'],
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def load_breed_data():
    """Load breed mapping data."""
    with open(BREED_MAPPING_FILE, 'r') as f:
        return json.load(f)


def initialize_rag_engine(model_type, chunking_strategy, use_reranking):
    """Initialize or reinitialize RAG engine."""
    with st.spinner("Initializing RAG engine..."):
        engine = RAGEngine(
            model_type=model_type,
            use_reranking=use_reranking,
            chunking_strategy=chunking_strategy,
        )

        # Ingest documents if needed
        if engine.collection.count() == 0:
            with st.spinner("Ingesting documents into vector database..."):
                doc_count = engine.ingest_documents()
                st.success(f"✓ Ingested {doc_count} documents")
        else:
            st.info(f"✓ Using existing vector database ({engine.collection.count()} documents)")

        return engine


def main():
    """Main Streamlit app."""

    # Header
    st.title(UI_CONFIG['title'])
    st.markdown(f"**{UI_CONFIG['subtitle']}**")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Check Ollama status
        ollama_status = check_ollama_available()
        if ollama_status:
            st.success("✓ Ollama server is running")
        else:
            st.error("✗ Ollama server not detected")
            st.warning("Please start Ollama: `ollama serve`")

        st.divider()

        # Model selection
        st.subheader("RAG Configuration")
        model_type = st.selectbox(
            "Model Type",
            options=list(MODELS.keys()),
            format_func=lambda x: MODELS[x]['description'],
            help="Choose between text-only and multimodal models"
        )

        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            options=list(CHUNKING_STRATEGIES.keys()),
            format_func=lambda x: CHUNKING_STRATEGIES[x]['description'],
            help="How to split documents into chunks"
        )

        use_reranking = st.checkbox(
            "Enable Reranking",
            value=True,
            help="Use cross-encoder to rerank retrieved documents"
        )

        st.divider()

        # Initialize button
        if st.button("Initialize RAG Engine", type="primary"):
            st.session_state.rag_engine = initialize_rag_engine(
                model_type, chunking_strategy, use_reranking
            )

        # Show current engine status
        if st.session_state.rag_engine:
            st.success("✓ RAG Engine Ready")
            stats = st.session_state.rag_engine.get_collection_stats()
            st.metric("Documents", stats['total_documents'])
            st.metric("Unique Breeds", stats['unique_breeds'])

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Query",
        "📊 Evaluation",
        "📈 Visualizations",
        "📖 Dataset Info"
    ])

    # Tab 1: Query Interface
    with tab1:
        st.header("Query Dog Breed Database")

        if not st.session_state.rag_engine:
            st.warning("⚠️ Please initialize the RAG engine from the sidebar first.")
        else:
            # Query input
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input(
                    "Ask a question about dog breeds:",
                    placeholder="e.g., What are good dog breeds for families with children?"
                )
            with col2:
                query_button = st.button("Search", type="primary")

            # Sample questions
            st.markdown("**Sample questions:**")
            sample_queries = [
                "What are the characteristics of Golden Retrievers?",
                "Which dog breeds are hypoallergenic?",
                "Compare Beagle and Chihuahua",
                "What breeds are good with children?",
                "Tell me about German Shepherds",
            ]

            cols = st.columns(len(sample_queries))
            for i, sample in enumerate(sample_queries):
                if cols[i].button(f"📝 {sample[:25]}...", key=f"sample_{i}"):
                    query = sample
                    query_button = True

            # Execute query
            if query_button and query:
                with st.spinner("🔍 Searching and generating answer..."):
                    start_time = time.time()
                    result = st.session_state.rag_engine.query(query)
                    elapsed_time = time.time() - start_time

                    # Save to history
                    st.session_state.query_history.append({
                        'query': query,
                        'result': result,
                        'timestamp': time.time(),
                    })

                    # Display results
                    st.divider()
                    st.subheader("Answer")
                    st.markdown(result['answer'])

                    # Metadata
                    st.divider()
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Response Time", f"{elapsed_time:.2f}s")
                    col2.metric("Model", result['model'])
                    col3.metric("Docs Retrieved", result['retrieval_count'])
                    col4.metric("Images Used", result.get('images_used', 0))

                    # Retrieved contexts
                    with st.expander("📚 Retrieved Contexts"):
                        for i, doc in enumerate(result['context_documents'], 1):
                            st.markdown(f"**Document {i}** (Breed: {doc['metadata']['breed']})")
                            st.text(doc['content'][:300] + "...")
                            if 'rerank_score' in doc:
                                st.caption(f"Similarity: {doc['similarity']:.3f} | Rerank Score: {doc['rerank_score']:.3f}")
                            else:
                                st.caption(f"Similarity: {doc['similarity']:.3f}")
                            st.divider()

            # Query history
            if st.session_state.query_history:
                st.divider()
                st.subheader("Query History")
                for i, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
                    with st.expander(f"{i}. {item['query'][:60]}..."):
                        st.markdown(f"**Answer:** {item['result']['answer'][:200]}...")

    # Tab 2: Evaluation
    with tab2:
        st.header("RAGAS Evaluation")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            Evaluate the RAG system using **RAGAS metrics**:
            - **Faithfulness**: Is the answer grounded in retrieved context?
            - **Answer Relevancy**: Does the answer address the question?
            - **Context Precision**: Were the right documents retrieved?
            - **Context Recall**: Were all relevant documents found?
            """)

        with col2:
            # Check if eval dataset exists
            eval_file = EVAL_DIR / "eval_dataset.json"
            if eval_file.exists():
                st.success("✓ Evaluation dataset ready")
            else:
                st.warning("⚠️ No evaluation dataset")

        st.divider()

        # Generate evaluation dataset
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Generate Evaluation Dataset")
            num_samples = st.number_input(
                "Number of Q&A pairs",
                min_value=10,
                max_value=100,
                value=30,
                step=10
            )

            if st.button("Generate Dataset"):
                with st.spinner("Generating evaluation dataset..."):
                    generator = EvalDatasetGenerator()
                    dataset = generator.generate_dataset(num_samples)
                    generator.save_dataset(dataset)
                    st.success(f"✓ Generated {len(dataset)} Q&A pairs")

        with col2:
            st.subheader("2. Run Evaluation")

            if not st.session_state.rag_engine:
                st.warning("⚠️ Initialize RAG engine first")
            elif not eval_file.exists():
                st.warning("⚠️ Generate evaluation dataset first")
            else:
                eval_samples = st.number_input(
                    "Samples to evaluate",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Fewer samples = faster evaluation"
                )

                if st.button("Run Evaluation"):
                    with st.spinner("Running RAGAS evaluation (this may take a few minutes)..."):
                        # Load dataset
                        with open(eval_file, 'r') as f:
                            eval_dataset = json.load(f)

                        # Run evaluation
                        evaluator = RAGEvaluator(st.session_state.rag_engine)
                        results = evaluator.evaluate_dataset(eval_dataset, limit=eval_samples)

                        # Store results
                        strategy_name = f"{model_type}_{chunking_strategy}_rerank{use_reranking}"
                        st.session_state.evaluation_results[strategy_name] = results

                        st.success("✓ Evaluation complete!")

        # Display results
        if st.session_state.evaluation_results:
            st.divider()
            st.subheader("Evaluation Results")

            # Create visualizer
            visualizer = RAGVisualizer()

            # Show radar chart
            radar_fig = visualizer.create_ragas_radar_chart(
                st.session_state.evaluation_results
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            # Show bar chart
            bar_fig = visualizer.create_metrics_bar_chart(
                st.session_state.evaluation_results
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            # Show comparison table
            comparison_df = visualizer.create_comparison_table(
                st.session_state.evaluation_results
            )
            st.dataframe(comparison_df, use_container_width=True)

    # Tab 3: Visualizations
    with tab3:
        st.header("Embedding Space Visualization")

        if not st.session_state.rag_engine:
            st.warning("⚠️ Please initialize the RAG engine first.")
        else:
            st.markdown("""
            Visualize the embedding space showing how dog breeds cluster based on their characteristics.
            Similar breeds will appear closer together in the 2D projection.
            """)

            # Options
            col1, col2 = st.columns(2)
            with col1:
                viz_method = st.selectbox(
                    "Visualization Method",
                    options=["UMAP", "t-SNE"],
                    help="Dimensionality reduction technique"
                )
            with col2:
                max_breeds = st.slider(
                    "Number of breeds to visualize",
                    min_value=10,
                    max_value=64,
                    value=30,
                    step=5
                )

            if st.button("Generate Embedding Map"):
                with st.spinner("Creating embedding visualization..."):
                    # Get documents from collection
                    results = st.session_state.rag_engine.collection.get(
                        limit=max_breeds,
                        include=['embeddings', 'metadatas', 'documents']
                    )

                    # Extract embeddings and labels
                    import numpy as np
                    embeddings = np.array(results['embeddings'])
                    labels = [meta.get('breed', 'Unknown') for meta in results['metadatas']]

                    # Create visualization
                    visualizer = RAGVisualizer()
                    embedding_fig = visualizer.create_embedding_space_map(
                        embeddings=embeddings,
                        labels=labels,
                        method=viz_method.lower(),
                        title=f"Dog Breed Embedding Space ({viz_method})",
                        metadata=results['metadatas']
                    )

                    st.plotly_chart(embedding_fig, use_container_width=True)

                    st.info(f"""
                    This visualization shows {len(labels)} dog breeds in 2D space based on their embeddings.
                    Breeds with similar characteristics should cluster together.
                    """)

    # Tab 4: Dataset Info
    with tab4:
        st.header("Dataset Information")

        # Load breed data
        breed_data = load_breed_data()

        st.markdown(f"""
        ### Dataset Overview
        - **Total Breeds**: {breed_data['total_breeds']}
        - **Total Images**: {sum(b['images_count'] for b in breed_data['breeds'])}
        - **Chunking Strategies**: {len(CHUNKING_STRATEGIES)}
        - **Available Models**: {len(MODELS)}
        """)

        st.divider()

        # Breed browser
        st.subheader("Browse Dog Breeds")

        # Create DataFrame
        breed_df = pd.DataFrame([
            {
                'Breed': b['name'],
                'Images': b['images_count'],
                'Country': b['characteristics'].get('Country of Origin', 'Unknown'),
                'Temperament': b['characteristics'].get('Character Traits', 'Unknown')[:50] + '...',
            }
            for b in breed_data['breeds']
        ])

        # Display table
        st.dataframe(breed_df, use_container_width=True)

        # Breed details
        st.divider()
        st.subheader("Breed Details")

        selected_breed = st.selectbox(
            "Select a breed to view details:",
            options=[b['name'] for b in breed_data['breeds']]
        )

        if selected_breed:
            breed_info = next(
                b for b in breed_data['breeds']
                if b['name'] == selected_breed
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"### {selected_breed}")
                chars = breed_info['characteristics']

                for key, value in chars.items():
                    st.markdown(f"**{key}:** {value}")

            with col2:
                st.markdown(f"### Image Gallery")
                st.info(f"Images available: {breed_info['images_count']}")
                st.caption(f"Folder: `{breed_info['image_folder']}`")


if __name__ == "__main__":
    main()
