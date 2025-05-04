"""
Created on Fri May 02 09:40:14 2025
@author: sid

Load and cache FastF1 session data for Race and Qualifying sessions
Also prepares feature-engineered datasets for training and prediction.
"""

import os
import shutil
import fastf1
import pandas as pd
from fastf1 import Cache, get_event_schedule

# Set base path
BASE_PATH = '/Users/sid/Downloads/F1_RacePredictions'
CACHE_PATH = os.path.join(BASE_PATH, 'cache')

# Enable and clean corrupted schedule cache
def reset_schedule_cache():
    sched_cache = os.path.join(CACHE_PATH, 'season_schedule')
    if os.path.exists(sched_cache):
        shutil.rmtree(sched_cache)
        print("✅ Deleted cached schedule files.")

# Enable FastF1 cache
def enable_cache():
    os.makedirs(CACHE_PATH, exist_ok=True)
    Cache.enable_cache(CACHE_PATH)

# Load and save session data (laps, results, weather)
def load_and_save_session(year, gp_name, session_type):
    try:
        print(f"⏳ Downloading {year} {gp_name} {session_type}...")
        session = fastf1.get_session(year, gp_name, session_type)
        session.load()

        folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
        os.makedirs(folder, exist_ok=True)

        session.laps.reset_index(drop=True).to_csv(os.path.join(folder, 'laps.csv'), index=False)
        session.results.reset_index(drop=True).to_csv(os.path.join(folder, 'results.csv'), index=False)
        session.weather_data.reset_index(drop=True).to_csv(os.path.join(folder, 'weather.csv'), index=False)

        print(f"✅ Saved {year} {gp_name} {session_type} data at: {folder}")
    except Exception as e:
        print(f"❌ Failed for {year} {gp_name} {session_type}: {e}")

# Fetch 2023 race and qualifying sessions
def fetch_2023_data():
    year = 2023
    try:
        schedule = get_event_schedule(year)
        gps = schedule['EventName'].tolist()
    except Exception as e:
        print(f"❌ Failed to load schedule: {e}")
        return

    for gp in gps:
        load_and_save_session(year, gp, 'R')  # Race
        load_and_save_session(year, gp, 'Q')  # Qualifying

if __name__ == '__main__':
    reset_schedule_cache()
    enable_cache()
    fetch_2023_data()
