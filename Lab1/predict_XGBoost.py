import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Define configuration settings
class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    FOLDER_PATH = ""
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_factors_with_direction.csv"
    VISUALIZATION_FILE = "learning_curve_xgboost.png"
    PREDICTION_PLOT_FILE = "predictions_vs_actual_xgboost.png"
    MODEL_FILE = "xgboost_model.json"
    
    # Train-test split parameters
    TEST_SIZE = 0.6
    
    # Training parameters
    N_ESTIMATORS = 1000
    LEARNING_RATE = 0.05
    MAX_DEPTH = 6
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    return pd.read_csv(file_path).dropna()

def prepare_features(df: pd.DataFrame):
    """Prepare features for XGBoost, excluding attributes containing 'target'."""
    
    if 'target_pct_change' not in df.columns:
        raise ValueError("❌ 'target_pct_change' column is missing from the dataset.")

    # Select numerical features except those containing "target" and timestamp
    feature_cols = [col for col in df.columns if "target" not in col and col != "open_time"]

    if not feature_cols:
        raise ValueError("❌ No valid features selected! Check the dataset structure.")

    print(f"✅ Selected {len(feature_cols)} features for training.")

    X = df[feature_cols].values
    y = df['target_pct_change'].values  

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, df

def train_xgboost(X, y, test_size=0.3):
    """Train an XGBoost model with Huber loss and return the trained model and evaluation results."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror",  # ✅ Correct objective for Huber-like loss
        n_estimators=Config.N_ESTIMATORS,
        learning_rate=Config.LEARNING_RATE,
        max_depth=Config.MAX_DEPTH,
        subsample=Config.SUBSAMPLE,
        colsample_bytree=Config.COLSAMPLE_BYTREE,
        eval_metric="rmse"  # Keep RMSE for evaluation tracking
    )


    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )

    evals_result = model.evals_result()

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ XGBoost Model MSE (Huber Loss): {mse:.4f}")

    return model, evals_result



def save_model(model, output_path: str):
    """Save the trained XGBoost model to a file."""
    model.save_model(output_path)
    print(f"✅ Model saved to {output_path}")

def save_predictions(df: pd.DataFrame, y_pred, output_path: str):
    """Save the predicted values to CSV file."""
    df['predicted_return'] = y_pred.flatten()
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

def plot_predictions(y_true, y_pred, output_path: str):
    """Plot actual vs. predicted values and save the figure."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", linestyle="solid", marker="o", markersize=3)
    plt.plot(y_pred, label="Predicted", linestyle="dashed", marker="s", markersize=3)
    plt.xlabel("Time Steps")
    plt.ylabel("Target Percentage Change")
    plt.title("Predicted vs Actual Values (XGBoost)")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Prediction plot saved to {output_path}")

def plot_learning_curve(evals_result, output_path: str):
    """Plot the learning curve for XGBoost and save the image."""
    train_rmse = evals_result["validation_0"]["rmse"]
    test_rmse = evals_result["validation_1"]["rmse"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse, label="Train RMSE", marker="o")
    plt.plot(test_rmse, label="Test RMSE", marker="s")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.title("XGBoost Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Learning curve saved to {output_path}")

def main():
    """Main execution function."""
    file_path = Config.INPUT_FILE
    output_path = Config.OUTPUT_FILE
    learning_curve_path = Config.VISUALIZATION_FILE
    prediction_plot_path = Config.PREDICTION_PLOT_FILE
    model_path = Config.MODEL_FILE

    df = load_data(file_path)
    X, y, scaler, df_filtered = prepare_features(df)

    model, evals_result = train_xgboost(X, y, test_size=Config.TEST_SIZE)

    save_model(model, model_path)

    y_pred = model.predict(X)
    save_predictions(df_filtered, y_pred, output_path)

    plot_predictions(y, y_pred, prediction_plot_path)
    plot_learning_curve(evals_result, learning_curve_path)

if __name__ == "__main__":
    main()
