"""
Created on Sun May 03 14:09:00 2025

@author: sid
Predict Miami 2025 Finishing Positions using trained model and simulated inputs
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings as w

# === Paths ===
BASE_PATH = r"/Users/sid/Downloads/F1_RacePredictions"
FEATURES_FILE = os.path.join(BASE_PATH, "2025_Miami Grand Prix_R", "features.csv")
DRIVER_FORM_FILE = os.path.join(BASE_PATH, "2025_Miami Grand Prix_R", "driver_form.csv")
MODEL_FILE = os.path.join(BASE_PATH, "race_result_regressor_optimized.pkl")
SCALER_FILE = os.path.join(BASE_PATH, "scaler_optimized.pkl")

w.filterwarnings('ignore')

IMPORTANT_FEATURES = [
    'QualiPosition',
    'PitStopCount',
    'AvgRaceLapTime',
    'AirTemp',
    'TrackTemp',
    'Humidity',
    'AvgFinishingPosition',
    'AvgQualifyingPosition'
]

# === Load model & scaler ===
def load_model_and_scaler():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

# === Load features ===
def load_feature_data():
    df = pd.read_csv(FEATURES_FILE)
    if 'FinalPosition' in df.columns:
        df.drop(columns=['FinalPosition'], inplace=True)  # prediction mode
    return df

# === Fill missing form data ===
def enrich_with_form(df):
    if os.path.exists(DRIVER_FORM_FILE):
        form = pd.read_csv(DRIVER_FORM_FILE)
        form.rename(columns={
            "AvgQualifyingPosition": "FallbackQuali",
            "AvgFinishingPosition": "FallbackFinish"
        }, inplace=True)
        df = df.merge(form, on="Driver", how="left")

        df['QualiPosition'] = df['QualiPosition'].fillna(df['FallbackQuali'])
        df['AvgFinishingPosition'] = df['AvgFinishingPosition'].fillna(df['FallbackFinish'])
        df['AvgQualifyingPosition'] = df['AvgQualifyingPosition'].fillna(df['FallbackQuali'])

    # Fallback to mid-grid if still missing
    df['QualiPosition'] = df['QualiPosition'].fillna(10)
    df['AvgFinishingPosition'] = df['AvgFinishingPosition'].fillna(10)
    df['AvgQualifyingPosition'] = df['AvgQualifyingPosition'].fillna(10)
    return df

# === Prediction ===
def make_predictions():
    print("📦 Loading model and scaler...")
    model, scaler = load_model_and_scaler()

    print("📄 Loading feature data for Miami 2025...")
    df = load_feature_data()
    df = enrich_with_form(df)

    print("🤖 Predicting finishing positions...")
    X = df[IMPORTANT_FEATURES].copy()
    X_scaled = scaler.transform(X)

    # Raw prediction
    df['PredictedScore'] = model.predict(X_scaled)

    # Assign integer ranks (1 = best)
    df['PredictedPosition'] = df['PredictedScore'].rank(method='min').astype(int)

    # Final output
    df = df[['Driver', 'QualiPosition', 'AvgFinishingPosition', 'AvgQualifyingPosition', 'PredictedPosition']]
    df.sort_values(by='PredictedPosition', inplace=True)

    print("\n🏁 Predicted Race Order - Miami 2025:")
    print(df.to_string(index=False))

    output_path = os.path.join(BASE_PATH, "2025_Miami Grand Prix_R", "predicted_positions.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved to: {output_path}")

# === Run ===
if __name__ == "__main__":
    make_predictions()