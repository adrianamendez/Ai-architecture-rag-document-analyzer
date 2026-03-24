"""
Visualization module for RAG evaluation and embedding analysis.
Creates RAGAS radar charts and embedding space maps.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

try:
    from umap import UMAP
except ImportError:
    UMAP = None

from sklearn.manifold import TSNE

from config import VIZ_CONFIG, RAGAS_METRICS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGVisualizer:
    """Create visualizations for RAG evaluation and analysis."""

    def __init__(self):
        """Initialize visualizer."""
        self.config = VIZ_CONFIG

    def create_ragas_radar_chart(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        title: str = "RAG Quality Metrics Comparison",
    ) -> go.Figure:
        """
        Create radar chart comparing RAGAS metrics across different RAG strategies.

        Args:
            evaluation_results: Dict mapping strategy names to evaluation results
            title: Chart title

        Returns:
            Plotly figure
        """
        # Extract metrics for each strategy
        strategies = []
        metrics_data = {metric: [] for metric in RAGAS_METRICS}

        for strategy_name, results in evaluation_results.items():
            strategies.append(strategy_name)
            scores = results.get('scores', {})

            for metric in RAGAS_METRICS:
                value = scores.get(metric, 0.0)
                metrics_data[metric].append(value)

        # Create radar chart
        fig = go.Figure()

        # Add trace for each strategy
        for i, strategy in enumerate(strategies):
            values = [metrics_data[metric][i] for metric in RAGAS_METRICS]
            # Close the radar by repeating first value
            values_closed = values + [values[0]]
            metrics_closed = RAGAS_METRICS + [RAGAS_METRICS[0]]

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=metrics_closed,
                fill='toself',
                name=strategy,
            ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                )
            ),
            showlegend=True,
            title=title,
            height=600,
        )

        return fig

    def create_metrics_bar_chart(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
        title: str = "RAG Metrics Comparison",
    ) -> go.Figure:
        """
        Create bar chart comparing metrics across strategies.

        Args:
            evaluation_results: Evaluation results for different strategies
            title: Chart title

        Returns:
            Plotly figure
        """
        # Prepare data
        data = []
        for strategy_name, results in evaluation_results.items():
            scores = results.get('scores', {})
            for metric in RAGAS_METRICS:
                data.append({
                    'Strategy': strategy_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': scores.get(metric, 0.0),
                })

        df = pd.DataFrame(data)

        # Create grouped bar chart
        fig = px.bar(
            df,
            x='Metric',
            y='Score',
            color='Strategy',
            barmode='group',
            title=title,
            labels={'Score': 'Score (0-1)', 'Metric': 'RAGAS Metric'},
            height=500,
        )

        fig.update_layout(yaxis_range=[0, 1])

        return fig

    def create_embedding_space_map(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        method: str = "umap",
        title: str = "Dog Breed Embedding Space",
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> go.Figure:
        """
        Create 2D visualization of embedding space using dimensionality reduction.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            labels: List of breed names for each embedding
            method: Reduction method ('umap' or 'tsne')
            title: Chart title
            metadata: Optional list of metadata dicts for hover info

        Returns:
            Plotly figure
        """
        logger.info(f"Creating embedding space visualization using {method.upper()}...")

        # Apply dimensionality reduction
        if method.lower() == "umap":
            if UMAP is None:
                logger.warning("UMAP not available, falling back to t-SNE")
                method = "tsne"
            else:
                reducer = UMAP(
                    n_components=self.config['embedding_map']['n_components'],
                    n_neighbors=self.config['embedding_map']['n_neighbors'],
                    min_dist=self.config['embedding_map']['min_dist'],
                    random_state=42,
                )
                coords = reducer.fit_transform(embeddings)

        if method.lower() == "tsne":
            reducer = TSNE(
                n_components=2,
                random_state=42,
                perplexity=min(30, len(embeddings) - 1),
            )
            coords = reducer.fit_transform(embeddings)

        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': coords[:, 0],
            'y': coords[:, 1],
            'breed': labels,
        })

        # Add metadata if available
        if metadata:
            for i, meta in enumerate(metadata):
                for key, value in meta.items():
                    if key not in df.columns:
                        df[key] = None
                    df.at[i, key] = value

        # Create scatter plot
        if metadata:
            # Color by a metadata field if available
            color_field = 'Country of Origin' if 'Country of Origin' in df.columns else 'breed'
        else:
            color_field = 'breed'

        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_field,
            hover_data=['breed'],
            title=title,
            labels={'x': f'{method.upper()} Dimension 1', 'y': f'{method.upper()} Dimension 2'},
            height=700,
        )

        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))

        return fig

    def create_breed_characteristics_radar(
        self,
        breed_data: Dict[str, Any],
        breed_name: str,
    ) -> go.Figure:
        """
        Create radar chart showing characteristics of a single breed.

        Args:
            breed_data: Breed characteristics
            breed_name: Name of the breed

        Returns:
            Plotly figure
        """
        # Map characteristics to numerical scores (0-10)
        char_mapping = {
            'size': self._map_size_to_score(breed_data.get('Height (in)', '')),
            'energy': 7,  # Default, could be extracted from traits
            'grooming': 5,  # Default
            'trainability': 8,  # Default
            'friendliness': self._map_temperament_to_score(
                breed_data.get('Character Traits', ''), 'friendly'
            ),
        }

        categories = list(char_mapping.keys())
        values = list(char_mapping.values())

        # Close the polygon
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name=breed_name,
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                )
            ),
            showlegend=True,
            title=f"{breed_name} Characteristics",
            height=500,
        )

        return fig

    def create_comparison_table(
        self,
        evaluation_results: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Create comparison table of RAG strategies.

        Args:
            evaluation_results: Evaluation results

        Returns:
            DataFrame with comparison
        """
        data = []

        for strategy_name, results in evaluation_results.items():
            scores = results.get('scores', {})

            row = {
                'Strategy': strategy_name,
                'Model': results.get('model', 'unknown'),
                'Chunking': results.get('chunking_strategy', 'unknown'),
                'Reranking': '✓' if results.get('reranking_used', False) else '✗',
                'Faithfulness': f"{scores.get('faithfulness', 0):.3f}",
                'Answer Relevancy': f"{scores.get('answer_relevancy', 0):.3f}",
                'Context Precision': f"{scores.get('context_precision', 0):.3f}",
                'Context Recall': f"{scores.get('context_recall', 0):.3f}",
                'Average': f"{scores.get('average', 0):.3f}",
            }
            data.append(row)

        df = pd.DataFrame(data)

        # Sort by average score
        df = df.sort_values('Average', ascending=False)

        return df

    def _map_size_to_score(self, height_str: str) -> float:
        """Map height string to 0-10 score."""
        try:
            # Extract first number from string like "21-24"
            height = int(height_str.split('-')[0]) if '-' in height_str else int(height_str)
            # Map to 0-10 scale (assuming 6-35 inch range)
            return min(10, max(0, (height - 6) / 3))
        except:
            return 5  # Default mid-range

    def _map_temperament_to_score(self, traits_str: str, keyword: str) -> float:
        """Map temperament trait presence to score."""
        if keyword.lower() in traits_str.lower():
            return 9
        elif 'good-natured' in traits_str.lower():
            return 7
        else:
            return 5

    def create_retrieval_performance_chart(
        self,
        retrieval_stats: List[Dict[str, Any]],
        title: str = "Retrieval Performance Comparison",
    ) -> go.Figure:
        """
        Create chart showing retrieval performance metrics.

        Args:
            retrieval_stats: List of retrieval statistics
            title: Chart title

        Returns:
            Plotly figure
        """
        df = pd.DataFrame(retrieval_stats)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Retrieval Time', 'Documents Retrieved'),
        )

        # Add traces
        if 'strategy' in df.columns and 'retrieval_time' in df.columns:
            fig.add_trace(
                go.Bar(x=df['strategy'], y=df['retrieval_time'], name='Time (s)'),
                row=1, col=1
            )

        if 'strategy' in df.columns and 'docs_retrieved' in df.columns:
            fig.add_trace(
                go.Bar(x=df['strategy'], y=df['docs_retrieved'], name='Documents'),
                row=1, col=2
            )

        fig.update_layout(height=400, title_text=title)

        return fig


def main():
    """Test visualizations."""
    print("=" * 60)
    print("Visualizer Test")
    print("=" * 60)

    visualizer = RAGVisualizer()

    # Create sample evaluation results
    print("\n1. Creating sample RAGAS radar chart...")
    sample_results = {
        'Text-Only RAG': {
            'scores': {
                'faithfulness': 0.85,
                'answer_relevancy': 0.78,
                'context_precision': 0.72,
                'context_recall': 0.68,
            }
        },
        'Multimodal RAG': {
            'scores': {
                'faithfulness': 0.88,
                'answer_relevancy': 0.82,
                'context_precision': 0.79,
                'context_recall': 0.75,
            }
        },
        'Multimodal + Reranking': {
            'scores': {
                'faithfulness': 0.92,
                'answer_relevancy': 0.89,
                'context_precision': 0.86,
                'context_recall': 0.81,
            }
        },
    }

    radar_fig = visualizer.create_ragas_radar_chart(sample_results)
    radar_fig.write_html('/tmp/ragas_radar_test.html')
    print("✓ Saved to /tmp/ragas_radar_test.html")

    # Create bar chart
    print("\n2. Creating metrics bar chart...")
    bar_fig = visualizer.create_metrics_bar_chart(sample_results)
    bar_fig.write_html('/tmp/metrics_bar_test.html')
    print("✓ Saved to /tmp/metrics_bar_test.html")

    # Create comparison table
    print("\n3. Creating comparison table...")
    comparison_df = visualizer.create_comparison_table(sample_results)
    print(comparison_df.to_string(index=False))

    # Create sample embedding space map
    print("\n4. Creating embedding space map...")
    # Generate random embeddings for testing
    n_breeds = 20
    embeddings = np.random.randn(n_breeds, 384)
    labels = [f"Breed_{i}" for i in range(n_breeds)]

    embedding_fig = visualizer.create_embedding_space_map(
        embeddings,
        labels,
        method='tsne',
    )
    embedding_fig.write_html('/tmp/embedding_space_test.html')
    print("✓ Saved to /tmp/embedding_space_test.html")

    print("\n" + "=" * 60)
    print("✓ Visualizer test complete!")
    print("Open the HTML files in your browser to view the charts.")
    print("=" * 60)


if __name__ == "__main__":
    main()
