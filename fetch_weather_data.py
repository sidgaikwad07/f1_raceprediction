"""
Created on Fri Apr 25 11:21:55 2025

@author: sid
"""
import pandas as pd
import fastf1
from fastf1 import get_session
import os

BASE_SAVE_PATH = r'/Users/sid/Downloads/F1_RacePredictions'

def fetch_weather_summary(year, gp_name, session_type='R'):
    cache_path = os.path.join(BASE_SAVE_PATH, "cache")
    os.makedirs(cache_path, exist_ok=True)
    fastf1.Cache.enable_cache(cache_path)
    
    print(f"Fetching weather data: {year} {gp_name} {session_type}")
    session = get_session(year, gp_name, session_type)
    session.load()
    weather_df = session.weather_data.reset_index(drop=True)
  
    weather_summary = weather_df[[
        'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall', 'WindSpeed', 'WindDirection']]
    summary_stats = weather_summary.describe().T.round(2)
    summary_stats.reset_index(inplace=True)
    summary_stats.rename(columns={'index': 'Feature'}, inplace=True)
    
    folder = os.path.join(BASE_SAVE_PATH, f"{year}_{gp_name}_{session_type}")
    os.makedirs(folder, exist_ok=True)
    summary_path = os.path.join(folder, "weather_summary.csv")
    summary_stats.to_csv(summary_path, index=False)

    print(f"Weather summary saved to {summary_path}")
    return weather_summary, summary_stats

if __name__ == "__main__":
    fetch_weather_summary(2025, 'Jeddah', 'R')
    fetch_weather_summary(2024, 'Miami', 'R')
   