"""
Created on Tue Apr 29 10:01:00 2025

@author: sid
Pole_to_Win Analysis :  Full 2023, 2024 seasons and dynamic 2025 races (Auto-Update till latest race)
With Heatmap Visualization
"""

import os
import fastf1
import pandas as pd
import warnings as w
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from pit_strategy_analysis import BASE_PATH

w.filterwarnings('ignore')

def load_or_fetch_results(year, gp_name, session_type='R'):
    folder = os.path.join(BASE_PATH, f"{year}_{gp_name}_{session_type}")
    results_path = os.path.join(folder, "results.csv")

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        return results_df
    else:
        print(f"📥 Fetching {year} {gp_name} results from FastF1...")
        try:
            session = fastf1.get_session(year, gp_name, session_type)
            session.load()

            os.makedirs(folder, exist_ok=True)
            results_df = session.results.reset_index(drop=True)
            results_df.to_csv(results_path, index=False)
            print(f"✅ Saved fetched results at {results_path}")
            return results_df
        except Exception as e:
            print(f"❌ Failed to fetch {year} {gp_name}: {e}")
            return None

def analyze_pole_to_win(year, gp_name):
    results = load_or_fetch_results(year, gp_name)
    if results is None:
        return None

    results = results[["Abbreviation", "GridPosition", "Position"]]

    if results.empty:
        print(f"⚠️ Empty results for {year} {gp_name}. Skipping...")
        return None

    pole_data = results.loc[results["GridPosition"] == 1]
    if pole_data.empty:
        print(f"⚠️ No pole sitter found for {year} {gp_name}. Skipping...")
        return None

    pole_sitter = pole_data["Abbreviation"].values[0]
    pole_finish = pole_data["Position"].values[0]

    won = 1 if pole_finish == 1 else 0

    return {
        "Year": year,
        "GrandPrix": gp_name,
        "PoleSitter": pole_sitter,
        "WonRace": bool(won)
    }

def get_all_grand_prix(year):
    schedule = fastf1.get_event_schedule(year)
    gps = schedule['EventName'].tolist()
    return gps

def get_completed_2025_races():
    """
    Fetch 2025 Grand Prix races that have already happened (by today's date).
    """
    schedule = fastf1.get_event_schedule(2025)
    today = datetime.utcnow()

    completed_races = []
    for idx, row in schedule.iterrows():
        if pd.isnull(row['Session5Date']):
            continue  # Skip if no race date
        if row['Session5Date'].to_pydatetime() < today:
            completed_races.append((2025, row['EventName']))
        else:
            break  # Future races, stop

    return completed_races

def pole_to_win_mixed_analysis(years_full, races_2025):
    full_records = []

    # Full seasons
    for year in years_full:
        gps = get_all_grand_prix(year)

        for gp in gps:
            record = analyze_pole_to_win(year, gp)
            if record:
                full_records.append(record)

    # Only completed races for 2025
    for year, gp in races_2025:
        record = analyze_pole_to_win(year, gp)
        if record:
            full_records.append(record)

    df = pd.DataFrame(full_records)

    if df.empty:
        print("\n❗ No race results found even after fetching. Please check FastF1 API or your race list.")
        return df, None

    # Calculate win rates per year
    win_rates = df.groupby("Year")["WonRace"].mean().multiply(100).reset_index()
    win_rates.rename(columns={"WonRace": "PoleToWinRate"}, inplace=True)

    print("\n🏆 Pole-to-Win Conversion Rates by Year:")
    print(win_rates)

    return df, win_rates

def plot_pole_to_win_heatmap(win_rates_df):
    heatmap_df = win_rates_df.pivot_table(index="Year", values="PoleToWinRate")

    plt.figure(figsize=(8, 4))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={'label': 'Pole-to-Win %'}
    )
    plt.title("Pole-to-Win Conversion Heatmap (2023–2025 till Latest Race)", fontsize=14)
    plt.tight_layout()

    output_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, "pole_to_win_heatmap_2023_to_2025_auto.png"))
    plt.show()

if __name__ == "__main__":
    # Full seasons
    full_years = [2023, 2024]

    # Auto-updating completed races for 2025
    races_2025 = get_completed_2025_races()
    print(f"\n📆 Completed 2025 races detected: {races_2025}\n")

    df, win_rates = pole_to_win_mixed_analysis(full_years, races_2025)
    print(df)

    if win_rates is not None:
        plot_pole_to_win_heatmap(win_rates)