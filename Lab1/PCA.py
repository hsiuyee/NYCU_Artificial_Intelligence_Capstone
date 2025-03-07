import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
class Config:
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_PCA.csv"
    EXPLAINED_VARIANCE_PLOT = "pca_explained_variance.png"
    N_COMPONENTS = 0.95  # Keep 95% variance

def load_data(file_path: str):
    """Load the dataset and remove non-numeric and target columns."""
    df = pd.read_csv(file_path)
    
    # Exclude columns that contain "target" (since they are labels)
    feature_cols = [col for col in df.columns if "target" not in col and df[col].dtype in [np.float64, np.int64]]
    
    print(f"✅ Initial feature count (excluding targets): {len(feature_cols)}")
    
    return df, df[feature_cols], feature_cols

def apply_pca(df_features, feature_names):
    """Apply PCA to reduce dimensionality while preserving variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Apply PCA
    pca = PCA(n_components=Config.N_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)

    # Number of selected components
    n_selected = pca.n_components_
    n_dropped = len(feature_names) - n_selected

    # Get retained feature names (sorted by explained variance)
    retained_features = np.array(feature_names)[np.argsort(-pca.explained_variance_ratio_)[:n_selected]]
    
    print(f"✅ PCA reduced dimensions to {n_selected}, preserving {Config.N_COMPONENTS * 100}% variance.")
    print(f"❌ Dropped {n_dropped} features (from {len(feature_names)} → {n_selected})")
    print(f"✅ Retained feature names: {list(retained_features)}")

    return X_pca, pca, n_selected, n_dropped, retained_features

def plot_explained_variance(pca):
    """Plot the cumulative explained variance ratio."""
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o", linestyle="--")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance Ratio")
    plt.grid()
    plt.savefig(Config.EXPLAINED_VARIANCE_PLOT)
    print(f"✅ Explained variance plot saved to {Config.EXPLAINED_VARIANCE_PLOT}")

def save_pca_data(df, X_pca, retained_features, output_path):
    """Save the PCA-transformed data along with original target labels."""
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

    # Add back the target columns (which were excluded from PCA)
    target_cols = [col for col in df.columns if "target" in col]
    df_pca[target_cols] = df[target_cols].iloc[len(df) - len(df_pca):].reset_index(drop=True)
    
    df_pca.to_csv(output_path, index=False)
    print(f"✅ PCA-transformed data saved to {output_path}")

def main():
    """Main function to execute PCA processing."""
    df, df_features, feature_names = load_data(Config.INPUT_FILE)
    X_pca, pca, n_selected, n_dropped, retained_features = apply_pca(df_features, feature_names)
    plot_explained_variance(pca)
    save_pca_data(df, X_pca, retained_features, Config.OUTPUT_FILE)

if __name__ == "__main__":
    main()
