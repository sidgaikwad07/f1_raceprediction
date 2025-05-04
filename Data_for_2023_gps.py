"""
Feature Engineering for 2023 season only.
Ensure all required files are downloaded for each race and qualifying session.
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

def get_driver_form_history(current_year, current_gp):
    history = []
    for year in range(2021, current_year):
        for folder in os.listdir(BASE_PATH):
            if folder.startswith(str(year)) and folder.endswith("_R"):
                try:
                    df = pd.read_csv(os.path.join(BASE_PATH, folder, "driver_form.csv"))
                    history.append(df)
                except:
                    continue
    if not history:
        return pd.DataFrame(columns=['Driver', 'AvgQualifyingPosition', 'AvgFinishingPosition'])
    all_history = pd.concat(history)
    return all_history.groupby("Driver")[["AvgQualifyingPosition", "AvgFinishingPosition"]].mean().reset_index()

def engineer_features_single_gp(year, gp_name, is_prediction=False):
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

    weather_summary = race_weather[["AirTemp", "TrackTemp", "Humidity"]].mean().round(2)
    for col in weather_summary.index:
        df[col] = weather_summary[col]

    df["GP"] = gp_name
    df["Year"] = year

    hist = get_driver_form_history(year, gp_name)
    df = df.merge(hist, on="Driver", how="left")

    save_path = os.path.join(BASE_PATH, f"{year}_{gp_name}_R")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "features.csv"), index=False)

    form = df[["Driver", "QualiPosition", "FinalPosition"]].rename(columns={
        "QualiPosition": "AvgQualifyingPosition",
        "FinalPosition": "AvgFinishingPosition"
    })
    form.to_csv(os.path.join(save_path, "driver_form.csv"), index=False)

    print(f"✅ Saved features and driver_form for {year} {gp_name}")
    return df

if __name__ == "__main__":
    races_2023 = [
        "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
        "Azerbaijan Grand Prix", "Miami Grand Prix", "Monaco Grand Prix",
        "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix",
        "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix",
        "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix",
        "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix",
        "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
        "Abu Dhabi Grand Prix"
    ]

    all_data = []
    for gp in races_2023:
        result = engineer_features_single_gp(2023, gp)
        if result is not None:
            all_data.append(result)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(BASE_PATH, "engineered_features_2023.csv"), index=False)
        print("📦 Combined 2023 feature dataset saved.")
    else:
        print("❌ No feature sets were created for 2023.")