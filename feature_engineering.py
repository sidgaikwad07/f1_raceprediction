"""
Created on Sat May 03 11:30:00 2025
@author: sid

Feature Engineering: Combine Race, Quali, Weather into model-friendly dataset
Supports both historical races and future race predictions
"""

import pandas as pd
import os
import fastf1
import warnings as w
import numpy as np
from fastf1 import plotting
w.filterwarnings('ignore')
fastf1.Cache.enable_cache('/Users/sid/Downloads/F1_RacePredictions/cache')
plotting.setup_mpl()

BASE_PATH = r'/Users/sid/Downloads/F1_RacePredictions'

def load_csvs(year, gp_name, session_type):
    folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
    laps_path = os.path.join(folder, "laps.csv")
    results_path = os.path.join(folder, "results.csv")
    weather_path = os.path.join(folder, "weather.csv")

    laps_df = pd.read_csv(laps_path)
    results_df = pd.read_csv(results_path)
    weather_df = pd.read_csv(weather_path)

    return laps_df, results_df, weather_df

def seconds_to_time_str(seconds):
    if pd.isnull(seconds):
        return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes}:{sec:02}.{millis:03}"

def load_driver_form(year, gp_name):
    path = os.path.join(BASE_PATH, f"{year}_{gp_name}_R", "driver_form.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=['Driver', 'AvgQualifyingPosition', 'AvgFinishingPosition'])

def engineer_features_single_gp(year, gp_name, is_prediction=False):
    if is_prediction:
        # For Miami 2025 prediction, use race weather + quali session only (no race results)
        race_results = pd.read_csv(os.path.join(BASE_PATH, f"{year}_{gp_name}_R", "results.csv"))
        race_weather = pd.read_csv(os.path.join(BASE_PATH, f"{year}_{gp_name}_R", "weather.csv"))
        quali_laps, quali_results, _ = load_csvs(year, gp_name, 'Q')
        race_laps = pd.read_csv(os.path.join(BASE_PATH, f"{year}_{gp_name}_R", "laps.csv"))
    else:
        race_laps, race_results, race_weather = load_csvs(year, gp_name, 'R')
        quali_laps, quali_results, _ = load_csvs(year, gp_name, 'Q')

    race_laps["LapTime"] = pd.to_timedelta(race_laps["LapTime"], errors='coerce')
    race_laps["LapTimeSec"] = race_laps["LapTime"].dt.total_seconds()

    avg_laps = race_laps.groupby("Driver")["LapTimeSec"].mean().reset_index()
    avg_laps.rename(columns={"LapTimeSec": "AvgRaceLapTime"}, inplace=True)
    avg_laps["ReadableAvgLap"] = avg_laps["AvgRaceLapTime"].apply(seconds_to_time_str)

    if not is_prediction:
        pit_counts = race_laps[race_laps["PitOutTime"].notna()].groupby("Driver").size().reset_index(name="PitStopCount")
    else:
        pit_counts = pd.DataFrame({"Driver": avg_laps["Driver"], "PitStopCount": 0})

    quali_positions = quali_results[["Abbreviation", "Position"]].rename(columns={"Abbreviation": "Driver", "Position": "QualiPosition"})

    if not is_prediction:
        race_positions = race_results[["Abbreviation", "Position"]].rename(columns={"Abbreviation": "Driver", "Position": "FinalPosition"})
    else:
        race_positions = pd.DataFrame({"Driver": avg_laps["Driver"], "FinalPosition": [None]*len(avg_laps)})

    features = avg_laps.merge(pit_counts, on="Driver", how="left")
    features = features.merge(quali_positions, on="Driver", how="left")
    features = features.merge(race_positions, on="Driver", how="left")

    weather_summary = race_weather[["AirTemp", "TrackTemp", "Humidity"]].mean().round(2)
    for col in weather_summary.index:
        features[col] = weather_summary[col]

    features["GP"] = gp_name
    features["Year"] = year

    historical_form = get_historical_driver_form(features, year, gp_name)
    features = features.merge(historical_form, on="Driver", how="left")

    save_folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_R")
    os.makedirs(save_folder, exist_ok=True)
    features.to_csv(os.path.join(save_folder, "features.csv"), index=False)
    print(f" Saved features.csv for {year} {gp_name}")

    create_driver_form(features, save_folder)
    return features

def create_driver_form(features_df, save_folder):
    form_data = features_df[['Driver', 'QualiPosition', 'FinalPosition']].copy()
    form_data.rename(columns={
        'QualiPosition': 'AvgQualifyingPosition',
        'FinalPosition': 'AvgFinishingPosition'
    }, inplace=True)
    form_data.to_csv(os.path.join(save_folder, "driver_form.csv"), index=False)
    print(f" Saved driver_form.csv")

def get_historical_driver_form(current_df, current_year, current_gp):
    history = []
    for year in range(2021, current_year):
        for folder in os.listdir(BASE_PATH):
            if folder.endswith("_R") and folder.startswith(str(year)):
                try:
                    df = pd.read_csv(os.path.join(BASE_PATH, folder, "driver_form.csv"))
                    df["Year"] = year
                    history.append(df)
                except Exception:
                    continue

    if not history:
        return pd.DataFrame({"Driver": current_df["Driver"], "AvgQualifyingPosition": np.nan, "AvgFinishingPosition": np.nan})

    all_history = pd.concat(history)
    form_avg = all_history.groupby("Driver")[['AvgQualifyingPosition', 'AvgFinishingPosition']].mean().reset_index()
    return form_avg