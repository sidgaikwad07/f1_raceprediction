"""
Combine engineered features from multiple years (2021–2025) into one dataset for training.
"""

import pandas as pd
import os

# Define base path and output file
BASE_PATH = '/Users/sid/Downloads/F1_RacePredictions'
OUTPUT_FILE = os.path.join(BASE_PATH, 'combined_engineered_features.csv')

# Input yearly feature files
feature_files = [
    "engineered_features_2021.csv",
    "engineered_features_2022.csv",
    "engineered_features_2023.csv",
    "engineered_features_2024.csv",
    "engineered_features_2025.csv"
]

# Columns required for model training
required_columns = [
    'QualiPosition', 'PitStopCount', 'AvgRaceLapTime',
    'AirTemp', 'TrackTemp', 'Humidity',
    'AvgFinishingPosition', 'AvgQualifyingPosition', 'FinalPosition'
]

# Combine all valid datasets
combined_data = []

for file_name in feature_files:
    file_path = os.path.join(BASE_PATH, file_name)
    try:
        df = pd.read_csv(file_path)
        # Ensure required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Skipping {file_name}: Missing columns {missing_cols}")
            continue

        # Ensure all required columns are numeric
        df[required_columns] = df[required_columns].apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing required values
        valid_df = df.dropna(subset=required_columns)

        combined_data.append(valid_df)
        print(f" Loaded {file_name}: {len(valid_df)} valid rows")

    except Exception as e:
        print(f" Error processing {file_name}: {e}")

# Save final combined dataset
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Combined dataset saved to: {OUTPUT_FILE}")
    print(f"🔢 Total training samples: {len(final_df)}")
else:
    print(" No valid data files to combine.")