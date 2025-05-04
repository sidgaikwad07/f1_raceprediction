"""
Created on Mon Apr 21 2025
Lap Time and Sector Comparison between Top Drivers
FastF1 styled plots with professional visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import fastf1
import fastf1.plotting

# Setup FastF1 plotting style
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=True, color_scheme='fastf1')

BASE_PATH = r"/Users/sid/Downloads/F1_RacePredictions"

# Top drivers we want to compare
Top_Drivers = ["VER", "PIA", "NOR", "RUS", "LEC"]

def load_laps(year, gp_name, session_type='R'):
    folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
    laps_path = os.path.join(folder, "laps.csv")
    laps_df = pd.read_csv(laps_path)
    return laps_df

def preprocess_laps(laps_df):
    # Convert lap and sector times to seconds
    laps_df["LapTime"] = pd.to_timedelta(laps_df["LapTime"], errors='coerce')
    laps_df["LapTimeSec"] = laps_df["LapTime"].dt.total_seconds()
    for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        laps_df[sector] = pd.to_timedelta(laps_df[sector], errors='coerce')
        laps_df[sector + "Sec"] = laps_df[sector].dt.total_seconds()
    return laps_df

def plot_lap_time_comparison(laps_df, session, year, gp_name):
    fig, ax = plt.subplots(figsize=(14, 7))

    for driver in Top_Drivers:
        driver_laps = laps_df[laps_df["Driver"] == driver]
        if not driver_laps.empty:
            team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else "Unknown"
            team_color = fastf1.plotting.get_team_color(team, session=session)

            ax.plot(driver_laps["LapNumber"], driver_laps["LapTimeSec"], label=driver, color=team_color)

    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title(f"Lap Time Comparison\n{session.event['EventName']} {year} {session.name}")
    ax.legend()
    ax.grid(True)

    output_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{year}_{gp_name}_lap_time_comparison.png"))
    plt.show()

def plot_sector_time_comparison(laps_df, session, year, gp_name):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    sectors = ["Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec"]
    titles = ["Sector 1 Time", "Sector 2 Time", "Sector 3 Time"]

    for idx, sector in enumerate(sectors):
        for driver in Top_Drivers:
            driver_laps = laps_df[laps_df["Driver"] == driver]
            if not driver_laps.empty:
                team = driver_laps['Team'].iloc[0] if 'Team' in driver_laps.columns else "Unknown"
                team_color = fastf1.plotting.get_team_color(team, session=session)

                axes[idx].plot(driver_laps["LapNumber"], driver_laps[sector], label=driver, color=team_color)

        axes[idx].set_ylabel("Sector Time (seconds)")
        axes[idx].set_title(f"{titles[idx]} Comparison")
        axes[idx].grid(True)

    axes[2].set_xlabel("Lap Number")
    plt.suptitle(f"Sector Time Comparison\n{session.event['EventName']} {year} {session.name}", fontsize=16)
    axes[0].legend()

    output_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{year}_{gp_name}_sector_time_comparison.png"))
    plt.show()

def analyze_driver_comparison(year, gp_name):
    # Load FastF1 session for metadata (event name, year, etc.)
    session = fastf1.get_session(year, gp_name, 'R')
    session.load()

    # Load locally saved laps data
    laps = load_laps(year, gp_name)
    laps = preprocess_laps(laps)

    # Create plots
    plot_lap_time_comparison(laps, session, year, gp_name)
    plot_sector_time_comparison(laps, session, year, gp_name)

if __name__ == "__main__":
    analyze_driver_comparison(2025, "Jeddah")
    analyze_driver_comparison(2024, "Miami")
