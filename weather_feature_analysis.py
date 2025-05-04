"""
Created on Mon Apr 28 11:48:18 2025

@author: sid
Weather Feature Analysis : Correlate weather conditions to lap time performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_PATH = r"/Users/sid/Downloads/F1_RacePredictions"

def load_laps_and_weather(year, gp_name, session_type='R'):
    folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
    laps = pd.read_csv(os.path.join(folder, "laps.csv"))
    weather = pd.read_csv(os.path.join(folder,"weather.csv"))
    return laps, weather

def preprocess_laps(laps_df):
    # Convert laptime to seconds
    laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTime"], errors='coerce')
    laps_df["LapTimeSec"] = laps_df["LapTime"].dt.total_seconds()
    return laps_df

def merge_laps_weather(laps_df, weather_df):
    # Simple merge on closest time available
    laps_df['Time'] = pd.to_timedelta(laps_df['Time'], errors='coerce')
    weather_df['Time'] = pd.to_timedelta(weather_df['Time'], errors='coerce')

    merged = pd.merge_asof(laps_df.sort_values(by='Time'),weather_df.sort_values(by='Time'),
        on='Time', direction='nearest')
    return merged


def plot_weather_vs_laptime(merged_df, year, gp_name):
    plt.figure(figsize=(16, 5))
    features = ["AirTemp", "TrackTemp", "Humidity"]
    for idx, feature in enumerate(features, 1):
        plt.subplot(1, 3, idx)
        sns.scatterplot(x=feature, y="LapTimeSec", data=merged_df)
        plt.title(f"{feature} vs Lap Time")
        plt.xlabel(feature)
        plt.ylabel("Lap Time(s)")
        plt.grid(True)
        
    plt.suptitle(f"Weather vs Lap Time Correlation - {gp_name} {year}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{year}_{gp_name}_weather_correlation.png"))
    plt.show()


def analyze_weather_impact(year, gp_name):
    laps, weather = load_laps_and_weather(year, gp_name)
    laps = preprocess_laps(laps)
    merged = merge_laps_weather(laps, weather)
    plot_weather_vs_laptime(merged, year, gp_name)
    
    # Calculate correlation coefficients
    corr = merged[["LapTimeSec", "AirTemp", "TrackTemp", "Humidity"]].corr()
    print(f"\n Correlation matrix for {gp_name} {year}:\n")
    print(corr["LapTimeSec"].sort_values(ascending=False))

if __name__ == "__main__":
    analyze_weather_impact(2025, "Jeddah")
    analyze_weather_impact(2024, "Miami")