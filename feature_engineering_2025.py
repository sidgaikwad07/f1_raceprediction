"""
Feature Engineering for 2025 Grand Prix (up to Miami).
Uses uploaded CSVs for Miami (qualifying available) and appends features.
"""

import os
import pandas as pd
import numpy as np
import fastf1
from fastf1 import plotting
import warnings
warnings.filterwarnings('ignore')

plotting.setup_mpl()

BASE_PATH = '/Users/sid/Downloads/F1_RacePredictions'

# ✅ Local full paths for Miami 2025
MIAMI_LAPS_PATH = '/Users/sid/Downloads/F1_RacePredictions/2025_Miami Grand Prix_Q/laps.csv'
MIAMI_WEATHER_PATH = '/Users/sid/Downloads/F1_RacePredictions/2025_Miami Grand Prix_Q/weather.csv'
MIAMI_RESULTS_PATH = '/Users/sid/Downloads/F1_RacePredictions/2025_Miami Grand Prix_Q/results.csv'

def get_historical_form():
    history = []
    for year in [2021, 2022, 2023, 2024]:
        path = os.path.join(BASE_PATH, f"engineered_features_{year}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df[['Driver', 'AvgQualifyingPosition', 'AvgFinishingPosition']]
            df['Year'] = year
            history.append(df)
    if not history:
        return pd.DataFrame(columns=['Driver', 'AvgQualifyingPosition', 'AvgFinishingPosition'])
    all_form = pd.concat(history)
    return all_form.groupby("Driver")[['AvgQualifyingPosition', 'AvgFinishingPosition']].mean().reset_index()

def engineer_miami_2025():
    print("🔧 Using uploaded Miami 2025 CSVs")
    historical_form = get_historical_form()

    # Load Miami qualifying session from FastF1
    session = fastf1.get_session(2025, 'Miami Grand Prix', 'Q')
    session.load()
    quali_results = session.results

    laps = pd.read_csv(MIAMI_LAPS_PATH)
    weather = pd.read_csv(MIAMI_WEATHER_PATH)
    results = pd.read_csv(MIAMI_RESULTS_PATH)

    drivers = quali_results['Abbreviation'].values
    df = pd.DataFrame({'Driver': drivers})

    # Add average lap time
    laps['LapTime'] = pd.to_timedelta(laps['LapTime'], errors='coerce')
    laps['LapTimeSec'] = laps['LapTime'].dt.total_seconds()
    avg_lap = laps.groupby("Driver")["LapTimeSec"].mean().reset_index(name="AvgRaceLapTime")
    pit_count = laps[laps['PitOutTime'].notna()].groupby("Driver").size().reset_index(name='PitStopCount')

    df = df.merge(avg_lap, on='Driver', how='left')
    df = df.merge(pit_count, on='Driver', how='left')
    df['PitStopCount'] = df['PitStopCount'].fillna(2).astype(int)

    # Quali results
    quali_pos = quali_results[['Abbreviation', 'Position']].rename(columns={
        'Abbreviation': 'Driver', 'Position': 'QualiPosition'
    })
    df = df.merge(quali_pos, on='Driver', how='left')

    # Final race results if any
    if not results.empty:
        final_pos = results[['Abbreviation', 'Position']].rename(columns={
            'Abbreviation': 'Driver', 'Position': 'FinalPosition'
        })
        df = df.merge(final_pos, on='Driver', how='left')
    else:
        df['FinalPosition'] = np.nan

    # Weather
    if not weather.empty:
        summary = weather[['AirTemp', 'TrackTemp', 'Humidity']].mean().round(2)
        for col in summary.index:
            df[col] = summary[col]
    else:
        df['AirTemp'] = np.nan
        df['TrackTemp'] = np.nan
        df['Humidity'] = np.nan

    df['Year'] = 2025
    df['GP'] = "Miami Grand Prix"
    df = df.merge(historical_form, on='Driver', how='left')

    save_path = os.path.join(BASE_PATH, "2025_Miami Grand Prix_R")
    os.makedirs(save_path, exist_ok=True)

    df.to_csv(os.path.join(save_path, "features.csv"), index=False)
    form_df = df[['Driver', 'QualiPosition', 'FinalPosition']].rename(columns={
        'QualiPosition': 'AvgQualifyingPosition',
        'FinalPosition': 'AvgFinishingPosition'
    })
    form_df.to_csv(os.path.join(save_path, "driver_form.csv"), index=False)

    print("✅ Saved features for 2025 Miami Grand Prix")

if __name__ == "__main__":
    engineer_miami_2025()