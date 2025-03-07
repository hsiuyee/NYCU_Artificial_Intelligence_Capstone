import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from hmmlearn.hmm import GaussianHMM

# Define configuration settings
class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_clusters_hmm.csv"
    CLUSTER_PLOT_FILE = "hmm_clusters.png"
    
    # HMM-GMM clustering parameters
    N_HIDDEN_STATES = 5  # Hidden states (市場狀態數量)
    MAX_ITER = 500  # HMM 訓練迭代次數
    RANDOM_STATE = 42  # 保證每次結果相同
    PCA_COMPONENTS = 0.87  # 保留 95% 資訊量

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    df = pd.read_csv(file_path).dropna()
    return df

def prepare_features(df: pd.DataFrame):
    """Prepare time-series features for HMM-GMM clustering."""
    
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

    # 降維 (保留 95% 變異數)
    pca = PCA(n_components=Config.PCA_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    print(f"✅ 原始維度: {X_scaled.shape[1]}, PCA 降維後: {X_pca.shape[1]}")

    return X_pca, scaler, df

def run_hmm_gmm(X_pca):
    """Run HMM-GMM clustering on time-series data."""
    print(f"🔹 Running HMM-GMM on shape: {X_pca.shape}")

    # 訓練 HMM-GMM
    hmm = GaussianHMM(n_components=Config.N_HIDDEN_STATES, covariance_type="full", 
                      random_state=Config.RANDOM_STATE, n_iter=Config.MAX_ITER)
    
    hmm.fit(X_pca)
    hidden_states = hmm.predict(X_pca)

    # 印出隱藏狀態分佈
    unique, counts = np.unique(hidden_states, return_counts=True)
    print("🔹 Hidden State Distribution:", dict(zip(unique, counts)))

    return hidden_states, hmm

def evaluate_clustering(X_pca, hidden_states, true_labels=None):
    """Evaluate the clustering performance with internal and external metrics."""
    
    print("\n🔹 Clustering Evaluation Metrics:")
    
    # Internal metrics
    silhouette_avg = silhouette_score(X_pca, hidden_states)
    print(f"✅ Silhouette Score: {silhouette_avg:.4f} (higher is better)")

    if true_labels is not None:
        # 確保 `true_labels` 和 `hidden_states` 具有相同長度
        if len(true_labels) != len(hidden_states):
            print("⚠️ Warning: `target_multiclass` and `hidden_states` have different lengths!")
            true_labels = true_labels[:len(hidden_states)]

        # External metrics
        ari = adjusted_rand_score(true_labels, hidden_states)
        nmi = normalized_mutual_info_score(true_labels, hidden_states)
        accuracy = accuracy_score(true_labels, hidden_states)

        print(f"✅ Adjusted Rand Index (ARI): {ari:.4f} (higher is better)")
        print(f"✅ Normalized Mutual Information (NMI): {nmi:.4f} (higher is better)")
        print(f"✅ Accuracy: {accuracy:.4f} (higher is better)")

    return silhouette_avg

def plot_clusters(hidden_states, output_path: str):
    """Visualize the distribution of hidden states."""
    
    unique_states, counts = np.unique(hidden_states, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique_states, counts, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Hidden State")
    plt.ylabel("Number of Samples")
    plt.title("HMM-GMM Hidden State Distribution")
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Cluster plot saved to {output_path}")

def save_clusters(df: pd.DataFrame, hidden_states, output_path: str):
    """Save the clustering results to a CSV file."""
    df = df.iloc[:len(hidden_states)].copy()  # 確保長度匹配
    df['hidden_state'] = hidden_states
    df.to_csv(output_path, index=False)
    print(f"✅ Hidden states saved to {output_path}")

def main():
    """Main execution function."""
    file_path = Config.INPUT_FILE
    output_path = Config.OUTPUT_FILE
    cluster_plot_path = Config.CLUSTER_PLOT_FILE

    df = load_data(file_path)
    X_pca, scaler, df_filtered = prepare_features(df)

    hidden_states, hmm = run_hmm_gmm(X_pca)

    # 確保 `target_multiclass` 存在
    if 'target_multiclass' in df_filtered.columns:
        true_labels = df_filtered['target_multiclass'].values
    else:
        true_labels = None
        print("⚠️ Warning: `target_multiclass` column not found, external evaluation skipped.")

    # Evaluate clustering
    evaluate_clustering(X_pca, hidden_states, true_labels)

    save_clusters(df_filtered, hidden_states, output_path)
    plot_clusters(hidden_states, cluster_plot_path)

if __name__ == "__main__":
    main()
