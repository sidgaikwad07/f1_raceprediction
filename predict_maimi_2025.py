"""
Predict Finishing Positions for Miami 2025 Grand Prix (Pre-race).
"""

import os
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# === Paths ===
BASE_PATH = '/Users/sid/Downloads/F1_RacePredictions'
FEATURE_FILE = os.path.join(BASE_PATH, "2025_Miami Grand Prix_R", "features.csv")
MODEL_FILE = os.path.join(BASE_PATH, "race_result_regressor_v2.pkl")
SCALER_FILE = os.path.join(BASE_PATH, "scaler_v2.pkl")
HISTORICAL_FILE = os.path.join(BASE_PATH, "combined_engineered_features.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "2025_Miami_PredictedResults.csv")

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

def load_model_and_scaler():
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print(f"✅ Loaded model: {os.path.basename(MODEL_FILE)}")
    print(f"✅ Loaded scaler: {os.path.basename(SCALER_FILE)}")
    return model, scaler

def get_driver_form():
    df = pd.read_csv(HISTORICAL_FILE)
    return df.groupby("Driver")[['AvgQualifyingPosition', 'AvgFinishingPosition']].mean()

def predict_positions():
    print("🔍 Predicting Miami 2025 Finishing Positions...")

    if not os.path.exists(FEATURE_FILE):
        print(f"❌ Feature file not found: {FEATURE_FILE}")
        return

    df = pd.read_csv(FEATURE_FILE)
    form_avgs = get_driver_form()

    # Fill missing form values
    for i, row in df.iterrows():
        driver = row['Driver']
        if pd.isna(row.get('AvgQualifyingPosition')) and driver in form_avgs.index:
            df.at[i, 'AvgQualifyingPosition'] = form_avgs.at[driver, 'AvgQualifyingPosition']
        if pd.isna(row.get('AvgFinishingPosition')) and driver in form_avgs.index:
            df.at[i, 'AvgFinishingPosition'] = form_avgs.at[driver, 'AvgFinishingPosition']

    # Fill missing numeric features if needed
    df['PitStopCount'] = df['PitStopCount'].fillna(2)
    df[['AvgRaceLapTime', 'AirTemp', 'TrackTemp', 'Humidity']] = df[
        ['AvgRaceLapTime', 'AirTemp', 'TrackTemp', 'Humidity']
    ].fillna(df[['AvgRaceLapTime', 'AirTemp', 'TrackTemp', 'Humidity']].mean())

    # Drop rows with missing required features
    missing = df[important_features].isna().sum()
    if missing.any():
        print("⚠️ Still missing values after imputation:\n", missing)
        df = df.dropna(subset=important_features)
    if df.empty:
        print("❌ No valid rows to predict.")
        return

    model, scaler = load_model_and_scaler()
    X = scaler.transform(df[important_features])
    predictions = model.predict(X)

    results = df[['Driver']].copy()
    results['PredictedPosition'] = np.round(predictions, 1)
    results = results.sort_values(by='PredictedPosition').reset_index(drop=True)
    results['PredictedRank'] = results.index + 1

    print("\n📊 Top 10 Predicted Finishers for Miami 2025:")
    print(results.head(10))

    results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Prediction results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    predict_positions()