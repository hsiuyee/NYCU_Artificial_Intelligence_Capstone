import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import cross_val_score

# Define configuration settings
class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    FOLDER_PATH = ""
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_clusters.csv"
    CLUSTER_PLOT_FILE = "kmeans_clusters.png"
    
    # Clustering parameters
    TIME_STEPS = 60
    N_CLUSTERS = 4  # 分成 4 群
    RANDOM_STATE = 42  # 保證每次結果一樣

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    return pd.read_csv(file_path).dropna()

def prepare_features(df: pd.DataFrame):
    """Prepare time-series features for unsupervised learning."""
    
    # Select numerical features
    feature_cols = [col for col in df.columns if "target" not in col and col != "open_time"]

    # Validate feature selection (ensure we have at least one feature)
    if not feature_cols:
        raise ValueError("❌ No valid features selected! Check the dataset structure.")

    print(f"✅ Selected {len(feature_cols)} features for clustering.")

    # Extract feature matrix (X)
    X = df[feature_cols].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define time step length for time-series clustering
    time_steps = Config.TIME_STEPS

    # Ensure there are enough data points for sequence generation
    if len(X_scaled) <= time_steps:
        raise ValueError("❌ Not enough data to generate sequences. Increase dataset size.")

    # Create sequences for clustering
    X_sequences = np.array([X_scaled[i-time_steps:i] for i in range(time_steps, len(X_scaled))])

    # Flatten sequences for K-Means
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)

    print(f"✅ Prepared sequences for clustering: {X_flat.shape}")

    # Retain proper indexing from the original DataFrame
    df_filtered = df.iloc[time_steps:].copy()

    return X_flat, scaler, df_filtered

def run_kmeans(X):
    """Run K-Means clustering on time-series sequences."""
    print(f"🔹 Running K-Means on shape: {X.shape}")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Print cluster counts
    unique, counts = np.unique(clusters, return_counts=True)
    print("🔹 Cluster distribution:", dict(zip(unique, counts)))

    return clusters, kmeans

def evaluate_clustering(X, clusters, kmeans, true_labels=None):
    """Evaluate the clustering performance with internal and external metrics."""
    
    print("\n🔹 Clustering Evaluation Metrics:")
    
    # Internal metrics (do not require labels)
    silhouette_avg = silhouette_score(X, clusters)
    inertia = kmeans.inertia_
    
    print(f"✅ Silhouette Score: {silhouette_avg:.4f} (higher is better)")
    print(f"✅ Inertia (SSE): {inertia:.4f} (lower is better)")

    if true_labels is not None:
        # External metrics (require true labels)
        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        print(f"✅ Adjusted Rand Index (ARI): {ari:.4f} (higher is better)")
        print(f"✅ Normalized Mutual Information (NMI): {nmi:.4f} (higher is better)")

    return silhouette_avg, inertia

def plot_clusters(df, clusters, output_path: str):
    """Visualize the distribution of clusters."""
    
    unique_clusters, counts = np.unique(clusters, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique_clusters, counts, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Samples")
    plt.title("K-Means Cluster Distribution")
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Cluster plot saved to {output_path}")

def save_clusters(df: pd.DataFrame, clusters, output_path: str):
    """Save the clustering results to a CSV file."""
    df['cluster'] = clusters
    df.to_csv(output_path, index=False)
    print(f"✅ Cluster labels saved to {output_path}")

def main():
    """Main execution function."""
    file_path = Config.INPUT_FILE
    output_path = Config.OUTPUT_FILE
    cluster_plot_path = Config.CLUSTER_PLOT_FILE

    df = load_data(file_path)
    X, scaler, df_filtered = prepare_features(df)

    clusters, kmeans = run_kmeans(X)
    
    # Evaluate clustering
    true_labels = df_filtered.get('target_multiclass', None)  # 如果有真實標籤，可以用來評估
    evaluate_clustering(X, clusters, kmeans, true_labels)

    save_clusters(df_filtered, clusters, output_path)
    plot_clusters(df_filtered, clusters, cluster_plot_path)

if __name__ == "__main__":
    main()
