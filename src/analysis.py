import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

def analyze_errors(df, labels, true_labels, output_dir="outputs"):
    """
    Performs failure analysis on the clustering results.
    1. Identifies misclassified items (based on cluster majority vote).
    2. Generates a Confusion Matrix for the Top 20 classes.
    3. Saves outputs to CSV and PNG.
    """
    if true_labels is None:
        print("No ground truth labels provided. Skipping error analysis.")
        return
        
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Deep Dive Error Analysis ---")
    
    cluster_to_label = {}
    df_analysis = df.copy()
    df_analysis['cluster'] = labels
    df_analysis['true_label'] = true_labels
    
    valid_clusters = df_analysis[df_analysis['cluster'] != -1]
    
    for c_id in valid_clusters['cluster'].unique():
        in_cluster = valid_clusters[valid_clusters['cluster'] == c_id]
        mode_label = in_cluster['true_label'].mode()
        if not mode_label.empty:
            cluster_to_label[c_id] = mode_label[0]
        else:
            cluster_to_label[c_id] = "Unknown"
            
    df_analysis['predicted_label'] = df_analysis['cluster'].map(cluster_to_label).fillna("Noise")
    
    
    errors = df_analysis[df_analysis['predicted_label'] != df_analysis['true_label']]
    error_rate = len(errors) / len(df_analysis)
    print(f"   > Estimated User-Level Error Rate: {error_rate:.2%}")
    
    error_cols = ['true_label', 'predicted_label', 'cluster', 'Title'] 
    for c in ['ProductName', 'Brand']:
        if c in df_analysis.columns:
            error_cols.append(c)
            
    errors[error_cols].head(500).to_csv(f"{output_dir}/misclassified_examples.csv", index=False)
    print(f"   > Saved 500 misclassified examples to {output_dir}/misclassified_examples.csv")
    
    top_categories = df_analysis['true_label'].value_counts().head(20).index
    
    filtered_df = df_analysis[df_analysis['true_label'].isin(top_categories) & 
                              df_analysis['predicted_label'].isin(top_categories)]
    
    if len(filtered_df) > 0:
        cm = confusion_matrix(filtered_df['true_label'], filtered_df['predicted_label'], labels=top_categories)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=top_categories, yticklabels=top_categories)
        plt.xlabel('Predicted Label (Cluster Majority)')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Top 20 Categories)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        cm_path = f"{output_dir}/confusion_matrix.png"
        plt.savefig(cm_path)
        print(f"   > Saved Confusion Matrix to {cm_path}")
        plt.close()
    else:
        print("   > Not enough data in top 20 categories to plot confusion matrix.")

    return error_rate
