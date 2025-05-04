
"""
Feature Engineering for 2021 season only (with rolling averages).
This script computes AvgQualifyingPosition and AvgFinishingPosition using prior 2021 races.
"""

import pandas as pd
import os
import fastf1
import numpy as np
import warnings as w
from fastf1 import plotting

w.filterwarnings('ignore')
plotting.setup_mpl()

BASE_PATH = r'/Users/sid/Downloads/F1_RacePredictions'

# Initialize cumulative form tracker
driver_form_tracker = []

def load_csvs(year, gp_name, session_type):
    folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
    return (
        pd.read_csv(os.path.join(folder, "laps.csv")),
        pd.read_csv(os.path.join(folder, "results.csv")),
        pd.read_csv(os.path.join(folder, "weather.csv"))
    )

def seconds_to_time_str(seconds):
    if pd.isnull(seconds): return None
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{minutes}:{sec:02}.{millis:03}"

def get_driver_form_rolling(current_driver_df):
    if not driver_form_tracker:
        return pd.DataFrame({
            "Driver": current_driver_df["Driver"],
            "AvgQualifyingPosition": np.nan,
            "AvgFinishingPosition": np.nan
        })
    history = pd.concat(driver_form_tracker)
    return history.groupby("Driver")[["AvgQualifyingPosition", "AvgFinishingPosition"]].mean().reset_index()

def engineer_features(year, gp_name):
    try:
        race_laps, race_results, race_weather = load_csvs(year, gp_name, 'R')
        quali_laps, quali_results, quali_weather = load_csvs(year, gp_name, 'Q')
    except Exception as e:
        print(f"⚠️ Skipping {year} {gp_name}: missing session files - {e}")
        return None

    race_laps["LapTime"] = pd.to_timedelta(race_laps["LapTime"], errors='coerce')
    race_laps["LapTimeSec"] = race_laps["LapTime"].dt.total_seconds()

    avg_laps = race_laps.groupby("Driver")["LapTimeSec"].mean().reset_index()
    avg_laps.rename(columns={"LapTimeSec": "AvgRaceLapTime"}, inplace=True)
    avg_laps["ReadableAvgLap"] = avg_laps["AvgRaceLapTime"].apply(seconds_to_time_str)

    pit_counts = race_laps[race_laps["PitOutTime"].notna()].groupby("Driver").size().reset_index(name="PitStopCount")

    quali_positions = quali_results[["Abbreviation", "Position"]].rename(columns={"Abbreviation": "Driver", "Position": "QualiPosition"})
    race_positions = race_results[["Abbreviation", "Position"]].rename(columns={"Abbreviation": "Driver", "Position": "FinalPosition"})

    df = avg_laps.merge(pit_counts, on="Driver", how="left")
    df = df.merge(quali_positions, on="Driver", how="left")
    df = df.merge(race_positions, on="Driver", how="left")

    # Weather
    weather_summary = race_weather[["AirTemp", "TrackTemp", "Humidity"]].mean().round(2)
    for col in weather_summary.index:
        df[col] = weather_summary[col]

    df["GP"] = gp_name
    df["Year"] = year

    # Add rolling driver form (using prior races)
    form = get_driver_form_rolling(df)
    df = df.merge(form, on="Driver", how="left")

    # Save current race driver form to global history
    current_form = df[["Driver", "QualiPosition", "FinalPosition"]].copy()
    current_form.rename(columns={
        "QualiPosition": "AvgQualifyingPosition",
        "FinalPosition": "AvgFinishingPosition"
    }, inplace=True)
    driver_form_tracker.append(current_form)

    # Save outputs
    save_path = os.path.join(BASE_PATH, f"{year}_{gp_name}_R")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "features.csv"), index=False)
    current_form.to_csv(os.path.join(save_path, "driver_form.csv"), index=False)
    print(f"✅ Saved features and driver_form for {year} {gp_name}")
    return df

def generate_2021_features_with_rolling_form():
    races_2021 = [
        "Bahrain Grand Prix", "Emilia Romagna Grand Prix", "Portuguese Grand Prix",
        "Spanish Grand Prix", "Monaco Grand Prix", "Azerbaijan Grand Prix",
        "French Grand Prix", "Styrian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Russian Grand Prix",
        "Turkish Grand Prix", "United States Grand Prix", "Mexico City Grand Prix",
        "São Paulo Grand Prix", "Qatar Grand Prix", "Saudi Arabian Grand Prix",
        "Abu Dhabi Grand Prix"
    ]

    full_features = []
    for gp_name in races_2021:
        try:
            features = engineer_features(2021, gp_name)
            if features is not None:
                full_features.append(features)
        except Exception as e:
            print(f"⚠️ Failed for 2021 {gp_name}: {e}")

    if full_features:
        combined = pd.concat(full_features, ignore_index=True)
        output_path = os.path.join(BASE_PATH, "engineered_features_2021.csv")
        combined.to_csv(output_path, index=False)
        print(f"✅ Yearly feature dataset saved: {output_path}")
    else:
        print("❌ No datasets created for 2021!")

if __name__ == "__main__":
    generate_2021_features_with_rolling_form()

