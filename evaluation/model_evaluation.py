"""
Evaluate the trained race prediction model on specific Grand Prix feature files.
"""

import os
import joblib
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# === Paths ===
BASE_PATH = '/Users/sid/Downloads/F1_RacePredictions'
MODEL_FILE = os.path.join(BASE_PATH, "race_result_regressor_v2.pkl")
SCALER_FILE = os.path.join(BASE_PATH, "scaler_v2.pkl")
HISTORICAL_FEATURES = os.path.join(BASE_PATH, "combined_engineered_features.csv")

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
    print(f"✅ Using model: {os.path.basename(MODEL_FILE)}")
    print(f"✅ Using scaler: {os.path.basename(SCALER_FILE)}")
    return model, scaler

def load_driver_form():
    df = pd.read_csv(HISTORICAL_FEATURES)
    return df.groupby("Driver")[['AvgQualifyingPosition', 'AvgFinishingPosition']].mean()

def evaluate_model_on_race(year, gp_name, driver_form):
    print(f"\n🔍 Evaluating model for {year} {gp_name}...")

    race_file = os.path.join(BASE_PATH, f"{year}_{gp_name}_R", "features.csv")
    if not os.path.exists(race_file):
        print(f" Feature file missing: {race_file}")
        return

    df = pd.read_csv(race_file)
    if 'Driver' not in df.columns:
        print(f" No Driver column in {race_file}")
        return

    # Fill missing AvgQuali/Finish using historical averages
    for col in ['AvgQualifyingPosition', 'AvgFinishingPosition']:
        if col not in df.columns:
            df[col] = np.nan

    for i, row in df.iterrows():
        if pd.isna(row['AvgQualifyingPosition']) and row['Driver'] in driver_form.index:
            df.at[i, 'AvgQualifyingPosition'] = driver_form.loc[row['Driver'], 'AvgQualifyingPosition']
        if pd.isna(row['AvgFinishingPosition']) and row['Driver'] in driver_form.index:
            df.at[i, 'AvgFinishingPosition'] = driver_form.loc[row['Driver'], 'AvgFinishingPosition']

    # Drop rows with missing input or target features
    df = df.dropna(subset=important_features + ['FinalPosition'])
    if df.empty:
        print("⚠️ No valid rows to evaluate.")
        return

    X = df[important_features]
    y_true = df['FinalPosition']

    model, scaler = load_model_and_scaler()
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"📊 MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")

    results_df = df[['Driver']].copy()
    results_df['Actual'] = y_true
    results_df['Predicted'] = np.round(y_pred, 1)
    results_df['AbsError'] = np.abs(y_true - y_pred)

    print(results_df.sort_values(by='Actual').head(10))

if __name__ == "__main__":
    driver_form = load_driver_form()

    test_races = [
        (2024, "Miami"),
        (2025, "Jeddah"),
        (2025, "Miami")
    ]

    for year, gp in test_races:
        evaluate_model_on_race(year, gp, driver_form)