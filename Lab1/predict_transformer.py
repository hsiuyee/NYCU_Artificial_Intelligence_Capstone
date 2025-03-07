import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define configuration settings
class Config:
    """
    Configuration class to centralize all hyperparameters and settings.
    """
    # File paths and names
    FOLDER_PATH = ""
    INPUT_FILE = "klines_BTC_with_factors.csv"
    OUTPUT_FILE = "klines_BTC_factors_with_direction.csv"
    VISUALIZATION_FILE = "learning_curve_transformer.png"
    PREDICTION_PLOT_FILE = "predictions_vs_actual_transformer.png"
    MODEL_FILE = "transformer_model.h5"
    
    # Train-test split parameters
    TEST_SIZE = 0.6
    FORECAST_HORIZON = 1
    
    # Training parameters
    EPOCHS = 20
    BATCH_SIZE = 32
    TIME_STEPS = 60

def load_data(file_path: str) -> pd.DataFrame:
    """Load the CSV file and return a DataFrame."""
    return pd.read_csv(file_path).dropna()

def prepare_features(df: pd.DataFrame):
    """Prepare all available features for Transformer model, excluding attributes containing 'target'."""
    
    # Ensure target column exists
    if 'target_pct_change' not in df.columns:
        raise ValueError("❌ 'target_pct_change' column is missing from the dataset.")

    # Select all numerical features except those containing "target" and timestamp
    feature_cols = [col for col in df.columns if "target" not in col and col != "open_time"]

    # Validate feature selection (ensure we have at least one feature)
    if not feature_cols:
        raise ValueError("❌ No valid features selected! Check the dataset structure.")

    print(f"✅ Selected {len(feature_cols)} features for training.")

    # Extract feature matrix (X) and target variable (y)
    X = df[feature_cols].values
    y = df['target_pct_change'].values  # Prediction target

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define time step length for Transformer input
    time_steps = Config.TIME_STEPS

    # Create sequences for Transformer input
    X_transformer = np.array([X_scaled[i-time_steps:i] for i in range(time_steps, len(X_scaled))])
    y_transformer = y[time_steps:]

    # Retain proper indexing from the original DataFrame
    df_filtered = df.iloc[time_steps:].copy()

    return X_transformer, y_transformer, scaler, df_filtered

def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.2):
    """Transformer encoder block with dropout and L2 regularization."""
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    x_ff = Dense(ff_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    return x

def build_transformer_model(input_shape, num_layers=6):
    """Build Transformer model with multiple encoder layers."""
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.02)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='huber')
    return model

def train_transformer(X, y, test_size=0.3, epochs=20, batch_size=32):
    """Train a Transformer model and return the trained model and loss history."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = build_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    return model, history

def plot_predictions(y_true, y_pred, output_path: str):
    """Plot actual vs. predicted values and save the figure."""
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual", linestyle="solid", marker="o", markersize=3)
    plt.plot(y_pred, label="Predicted", linestyle="dashed", marker="s", markersize=3)
    plt.xlabel("Time Steps")
    plt.ylabel("Target Percentage Change")
    plt.title("Predicted vs Actual Values")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Prediction plot saved to {output_path}")

def plot_learning_curve(history, output_path: str):
    """Plot the learning curve for Transformer and save the image."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Test Loss', marker='s')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Transformer Learning Curve")
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    print(f"✅ Learning curve saved to {output_path}")

def save_model(model, output_path: str):
    """Save the trained model to a file."""
    model.save(output_path)
    print(f"✅ Model saved to {output_path}")

def save_predictions(df: pd.DataFrame, y_pred, output_path: str):
    """Save the predicted values to CSV file."""
    df['predicted_return'] = y_pred.flatten()
    df.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

def main():
    """Main execution function."""
    file_path = Config.INPUT_FILE
    output_path = Config.OUTPUT_FILE
    learning_curve_path = Config.VISUALIZATION_FILE
    prediction_plot_path = Config.PREDICTION_PLOT_FILE
    model_path = Config.MODEL_FILE

    df = load_data(file_path)
    X, y, scaler, df_filtered = prepare_features(df)

    model, history = train_transformer(X, y, test_size=Config.TEST_SIZE, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE)

    save_model(model, model_path)

    y_pred = model.predict(X)
    save_predictions(df_filtered, y_pred, output_path)

    plot_predictions(y, y_pred, prediction_plot_path)
    plot_learning_curve(history, learning_curve_path)

if __name__ == "__main__":
    main()