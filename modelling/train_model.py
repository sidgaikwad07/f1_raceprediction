"""
Train model to predict Finishing Position with the Current Driver Performance
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings as w
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

w.filterwarnings('ignore')

# === Paths ===
BASE_PATH = r"/Users/sid/Downloads/F1_RacePredictions"
FEATURES_FILE = os.path.join(BASE_PATH, "combined_engineered_features.csv")
MODEL_FILE = os.path.join(BASE_PATH, "race_result_regressor_v2.pkl")
SCALER_FILE = os.path.join(BASE_PATH, "scaler_v2.pkl")
IMAGE_FOLDER = os.path.join(BASE_PATH, "images")

# === Feature columns for training ===
important_features = [
    'QualiPosition',
    'PitStopCount',
    'AvgRaceLapTime',
    'AirTemp',
    'TrackTemp',
    'Humidity',
    'AvgFinishingPosition',
    'AvgQualifyingPosition'
]

# === Data Preprocessing ===
def preprocess_data(df, features):
    df = df.dropna(subset=features + ['FinalPosition'])
    df = df[df['FinalPosition'] <= 20]  # drop DNS/DNF etc.
    X = df[features]
    y = df['FinalPosition']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# === Model Training ===
def train_model(df):
    print("🧹 Preprocessing features...")
    X, y, scaler = preprocess_data(df, important_features)

    print("\n📊 FinalPosition target variable summary:")
    print(y.describe())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("🔍 Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2]
    }

    grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best parameters: {grid.best_params_}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print("\n📈 Regression Metrics:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.2f}")

    # Save feature importance plot
    feat_importance = pd.Series(best_model.feature_importances_, index=important_features)
    plt.figure(figsize=(8, 4))
    feat_importance.sort_values().plot(kind='barh', title='Feature Importance (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    plt.savefig(os.path.join(IMAGE_FOLDER, "feature_importance_v2.png"))
    plt.close()

    # Save model and scaler
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\n💾 New model saved to: {MODEL_FILE}")
    print(f"💾 New scaler saved to: {SCALER_FILE}")

# === Main Execution ===
if __name__ == "__main__":
    print("📥 Loading dataset...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"✅ Loaded {df.shape[0]} samples with {df.shape[1]} features.")
    train_model(df)