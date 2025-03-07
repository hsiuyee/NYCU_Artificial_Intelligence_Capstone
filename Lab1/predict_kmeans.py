import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

# Define configuration settings
class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_clusters.csv"
    CLUSTER_PLOT_FILE = "kmeans_clusters.png"
    
    # Clustering parameters
    TIME_STEPS = 60
    N_CLUSTERS = 5  # 分成 4 群
    RANDOM_STATE = 42  # 保證每次結果一樣

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    df = pd.read_csv(file_path).dropna()
    return df

def prepare_features(df: pd.DataFrame):
    """Prepare time-series features for unsupervised learning."""
    
    # 過濾掉 target 和時間欄位，避免影響特徵選擇
    feature_cols = [col for col in df.columns if "target" not in col and col != "open_time"]

    if not feature_cols:
        raise ValueError("❌ No valid features selected! Check the dataset structure.")

    print(f"✅ Selected {len(feature_cols)} features for clustering.")

    # 轉換為矩陣
    X = df[feature_cols].values

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 創建時間序列窗口
    time_steps = Config.TIME_STEPS
    if len(X_scaled) <= time_steps:
        raise ValueError("❌ Not enough data to generate sequences. Increase dataset size.")

    X_sequences = np.array([X_scaled[i-time_steps:i] for i in range(time_steps, len(X_scaled))])
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)

    print(f"✅ Prepared sequences for clustering: {X_flat.shape}")

    df_filtered = df.iloc[time_steps:].copy()

    return X_flat, scaler, df_filtered

def run_kmeans(X):
    """Run K-Means clustering on time-series sequences."""
    print(f"🔹 Running K-Means on shape: {X.shape}")

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=Config.N_CLUSTERS, random_state=Config.RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X)

    unique, counts = np.unique(clusters, return_counts=True)
    print("🔹 Cluster distribution:", dict(zip(unique, counts)))

    return clusters, kmeans

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

def compute_accuracy(true_labels, predicted_clusters):
    """Compute accuracy by mapping predicted clusters to true labels using the Hungarian algorithm."""
    if len(true_labels) != len(predicted_clusters):
        print("⚠️ Warning: Length mismatch between `target_multiclass` and predicted clusters. Adjusting.")
        min_len = min(len(true_labels), len(predicted_clusters))
        true_labels, predicted_clusters = true_labels[:min_len], predicted_clusters[:min_len]

    # Create a contingency matrix
    contingency_matrix = np.zeros((Config.N_CLUSTERS, Config.N_CLUSTERS), dtype=int)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], predicted_clusters[i]] += 1

    # Hungarian Algorithm for optimal label assignment
    row_ind, col_ind = linear_sum_assignment(contingency_matrix.max() - contingency_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Map predicted clusters to the best-matching true labels
    predicted_mapped = np.array([mapping[c] for c in predicted_clusters])
    accuracy = accuracy_score(true_labels, predicted_mapped)

    return accuracy


def evaluate_clustering(X, clusters, kmeans, true_labels=None):
    """Evaluate clustering performance using internal and external metrics."""
    
    print("\n🔹 Clustering Evaluation Metrics:")
    
    # Internal metrics
    silhouette_avg = silhouette_score(X, clusters)
    inertia = kmeans.inertia_
    
    print(f"✅ Silhouette Score: {silhouette_avg:.4f} (higher is better)")
    print(f"✅ Inertia (SSE): {inertia:.4f} (lower is better)")

    if true_labels is not None:
        if len(true_labels) != len(clusters):
            print("⚠️ Warning: `target_multiclass` and `clusters` have different lengths!")
            true_labels = true_labels[:len(clusters)]

        ari = adjusted_rand_score(true_labels, clusters)
        nmi = normalized_mutual_info_score(true_labels, clusters)
        accuracy = compute_accuracy(true_labels, clusters)

        print(f"✅ Adjusted Rand Index (ARI): {ari:.4f} (higher is better)")
        print(f"✅ Normalized Mutual Information (NMI): {nmi:.4f} (higher is better)")
        print(f"✅ Accuracy: {accuracy:.4f} (higher is better)")

    return silhouette_avg, inertia

def plot_clusters(clusters, output_path: str):
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
    df = df.iloc[:len(clusters)].copy()  # 確保長度匹配
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

    # 確保 `target_multiclass` 存在
    if 'target_multiclass' in df_filtered.columns:
        true_labels = df_filtered['target_multiclass'].values
    else:
        true_labels = None
        print("⚠️ Warning: `target_multiclass` column not found, external evaluation skipped.")

    # Evaluate clustering
    evaluate_clustering(X, clusters, kmeans, true_labels)

    save_clusters(df_filtered, clusters, output_path)
    plot_clusters(clusters, cluster_plot_path)

if __name__ == "__main__":
    main()
