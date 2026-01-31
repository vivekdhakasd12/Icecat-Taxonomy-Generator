import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score

def run_baseline(embeddings, labels, test_size=0.2, random_state=42):
    """
    Trains a Logistic Regression classifier to establish a Supervised Upper Bound.
    Returns metrics dict compatible with the clustering report.
    """
    print(f"\n--- Supervised Baseline (Logistic Regression) ---")
    print(f"   > Splitting data (Train={1-test_size:.0%}, Test={test_size:.0%})...")
    
    if labels.isnull().any():
        print("   > Warning: Labels contain NaNs. Dropping them for supervised training.")
        mask = ~labels.isnull()
        X = embeddings[mask]
        y = labels[mask]
    else:
        X = embeddings
        y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"   > Training Classifier (max_iter=1000)...")
    start = time.time()
    clf = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=random_state)
    clf.fit(X_train, y_train)
    
    print(f"   > Predicting on Test Set...")
    y_pred = clf.predict(X_test)
    elapsed = time.time() - start
    
    acc = accuracy_score(y_test, y_pred)
    nmi = normalized_mutual_info_score(y_test, y_pred)
    ari = adjusted_rand_score(y_test, y_pred)
    
    print(f"   > Supervised Baseline Completed in {elapsed:.2f}s.")
    print(f"   > Test Accuracy: {acc:.2%}")
    
    return {
        'purity': acc, # Accuracy is Purity in supervised context
        'nmi': nmi,
        'ari': ari,
        'silhouette': 0.0, # Not applicable really, or we could compute it but it's expensive
        'n_clusters': len(set(y)),
        'noise_ratio': 0.0,
        'best_params': "LogisticRegression(Default)"
    }
