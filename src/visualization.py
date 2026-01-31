import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from . import config

def reduce_dimensions(embeddings, method='pca', n_components=2):
    if method == 'pca':
        print("Reducing dimensions with PCA...")
        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        return pca.fit_transform(embeddings)
    elif method == 'tsne':
        print("Reducing dimensions with t-SNE...")
        tsne = TSNE(n_components=n_components, random_state=config.RANDOM_SEED, init='pca', learning_rate='auto')
        return tsne.fit_transform(embeddings)
    elif method == 'umap':
        try:
            import umap
            print("Reducing dimensions with UMAP...")
            reducer = umap.UMAP(n_components=n_components, random_state=config.RANDOM_SEED)
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("UMAP not installed. Falling back to PCA.")
            return reduce_dimensions(embeddings, method='pca', n_components=n_components)

def plot_clusters_2d(embeddings_2d, labels, title="Cluster Visualization", true_labels=None, interactive=True, output_path=None):
    """
    Plots the 2D embeddings colored by cluster label.
    """
    df_plot = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_plot['cluster'] = labels.astype(str)
    
    if true_labels is not None:
        df_plot['true_label'] = true_labels
        hover_data = ['cluster', 'true_label']
    else:
        hover_data = ['cluster']

    if interactive:
        fig = px.scatter(
            df_plot, x='x', y='y', color='cluster',
            title=title,
            hover_data=hover_data,
            width=800, height=600
        )
        if output_path:
            html_path = output_path.replace('.png', '.html')
            print(f"Saving interactive plot to {html_path}")
            fig.write_html(html_path)
        else:
            fig.show()
    else:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster', palette='tab10', s=10, alpha=0.6, legend='full')
        plt.title(title)
        if output_path:
             print(f"Saving static plot to {output_path}")
             plt.savefig(output_path)
             plt.close()
        else:
             plt.show()

def plot_comparison_panel(embeddings_2d, labels_dict, true_labels=None, output_path="outputs/clustering_comparison_panel.png"):
    """
    Creates a static subplot grid comparing multiple clustering results.
    """
    n_algorithms = len(labels_dict) + (1 if true_labels is not None else 0)
    cols = 3
    rows = (n_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    axes = axes.flatten() if n_algorithms > 1 else [axes]
    
    df_plot = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    
    plot_idx = 0
    if true_labels is not None:
        ax = axes[plot_idx]
        df_plot['true'] = true_labels
        sns.scatterplot(data=df_plot, x='x', y='y', hue='true', ax=ax, s=5, alpha=0.6, legend=False, palette='tab10')
        ax.set_title("Ground Truth (Categories)")
        ax.axis('off')
        plot_idx += 1
        
    for algo_name, labels in labels_dict.items():
        if plot_idx >= len(axes): break
        ax = axes[plot_idx]
        df_plot['cluster'] = labels.astype(str)
        sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster', ax=ax, s=5, alpha=0.6, legend=False, palette='tab10')
        ax.set_title(f"{algo_name} Clusters")
        ax.axis('off')
        plot_idx += 1
        
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Comparison Panel to {output_path}")
    plt.close()

def plot_metrics_comparison(results_summary, output_path="outputs/metrics_comparison.png"):
    """
    Plots a grouped bar chart comparing metrics (Purity, NMI, ARI) across algorithms.
    """
    df = pd.DataFrame(results_summary).T
    df = df.reset_index().rename(columns={'index': 'Algorithm'})
    
    start_cols = ['Algorithm']
    metric_cols = [c for c in ['purity', 'nmi', 'ari', 'silhouette'] if c in df.columns]
    
    if not metric_cols:
        print("No metrics to plot.")
        return

    df_melt = df.melt(id_vars=start_cols, value_vars=metric_cols, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Algorithm', y='Score', hue='Metric', palette='viridis')
    plt.title("Clustering Performance Comparison")
    plt.ylim(-0.1, 1.0)
    plt.axhline(0, color='grey', linewidth=0.8)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved Metrics Comparison to {output_path}")
    plt.close()

def plot_sankey_flow(df, true_col, pred_col, output_path="outputs/cluster_sankey.html", title="True Labels -> Clusters"):
    """
    Creates a Sankey diagram mapping True Categories to Predicted Clusters.
    Only shows Top 15 categories to avoid clutter.
    """
    top_categories = df[true_col].value_counts().head(15).index
    df_sub = df[df[true_col].isin(top_categories)].copy()
    
    
    labels_src = sorted(list(df_sub[true_col].unique()))
    labels_tgt = sorted(list(df_sub[pred_col].unique()))
    
    all_labels = labels_src + [f"Cluster {c}" for c in labels_tgt]
    
    src_map = {l: i for i, l in enumerate(labels_src)}
    tgt_map = {l: i + len(labels_src) for i, l in enumerate(labels_tgt)}
    
    flows = df_sub.groupby([true_col, pred_col]).size().reset_index(name='count')
    
    sources = flows[true_col].map(src_map).tolist()
    targets = flows[pred_col].map(tgt_map).tolist()
    values = flows['count'].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = all_labels,
          color = "blue"
        ),
        link = dict(
          source = sources,
          target = targets,
          value = values
      ))])

    fig.update_layout(title_text=title, font_size=10)
    fig.write_html(output_path)
    print(f"Saved Sankey Diagram to {output_path}")
