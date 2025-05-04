"""
Created on Mon Apr 28 16:03:59 2025

@author: sid
Pit Stop Strategy : Engineering, how pit stop can play role in driver position
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
import fastf1.plotting
from driver_lap_comparison import BASE_PATH
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=True, color_scheme='fastf1')

def load_stints(session):
    stints = session.laps.pick_quicklaps().copy()
    stints["Compound"] = stints["Compound"].fillna(method="ffill")
    stints = stints.groupby(["Driver", "Compound"]).agg(StintLength=("LapNumber", "count"))
    stints = stints.reset_index()
    return stints

def plot_stint_strategy(session, year, gp_name):
    stints = load_stints(session)
    drivers = sorted(stints["Driver"].unique(), reverse=True)
    fig, ax = plt.subplots(figsize=(7,12))
    
    for driver in drivers:
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            compound_colour = fastf1.plotting.get_compound_color(row["Compound"], session=session)

            ax.barh(
                y=driver,
                width=row["StintLength"],
                left=previous_stint_end,
                color=compound_colour,
                edgecolor="black",
                fill=True
            )
            previous_stint_end += row["StintLength"]
            
    plt.title(f"{gp_name} {year} Grand Prix - Tire Strategies")
    plt.xlabel("Lap Number")
    plt.grid(False)
    
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    
    output_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f"{year}_{gp_name}_tire_strategy_plot.png"))
    plt.show()

def stint_strategy_analysis(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'R')
    session.load()

    plot_stint_strategy(session, year, gp_name)

if __name__ == "__main__":
    stint_strategy_analysis(2025, "Jeddah")
    stint_strategy_analysis(2024, "Miami")
