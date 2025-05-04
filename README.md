# f1_raceprediction
Prediction for F1 Races

# 🏁 F1 Race Predictions using FastF1 and XGBoost

This project builds a predictive model to forecast **Formula 1 Grand Prix race results** using real-world qualifying performance, weather conditions, and historical driver form.

The current model accurately predicts **driver finishing positions**, demonstrated on the **Miami Grand Prix 2025** using the latest available qualifying data.

---

## 🚀 Project Highlights

- 📊 **Machine Learning Model**: Built using `XGBoost` regression to predict finishing positions
- 📡 **Real-Time Data Fetching**: Pulls official F1 session data (laps, weather, results) using `FastF1`
- 🌤️ **Weather Impact**: Integrates air temp, track temp, and humidity into prediction pipeline
- 🧠 **Driver Form Engineering**: Leverages multi-year stats to encode consistency and historical strength
- 🏎️ **Miami GP 2025 Predictions**: Published predictions based on actual qualifying performance

---

## 📂 Project Structure
F1_RacePredictions/
├── data/                        # Processed session data (laps, weather, results)
├── engineered_features_*.csv   # Feature sets from 2021–2025
├── combined_engineered_features.csv
├── prediction/                 # Miami 2025 prediction output
│   └── predicted_positions.csv
├── images/                     # Feature importance visualizations
├── feature_engineering_2025.py
├── train_model.py
├── model_evaluation.py
├── predict_miami_2025.py
└── README.md

## ⚙️ How It Works

### 1. 🛠 Feature Engineering
- Uses `FastF1` to load **qualifying**, **race**, and **weather** data
- Adds **PitStopCount**, **AvgRaceLapTime**, and driver’s **historical form**
- Extracts and saves features into year-specific CSVs

### 2. 🧪 Model Training
- Loads `combined_engineered_features.csv` (2021–2025)
- Trains `XGBoostRegressor` with hyperparameter tuning
- Evaluates performance using MAE, RMSE, and R²

### 3. 🔍 Race Evaluation
- Evaluates trained model on historical races like **Jeddah 2025** or **Miami 2024**
- Compares predicted and actual positions

### 4. 🏁 Miami 2025 Prediction
- Uses live `QualiPosition` and historical form
- Assumes pit stops and weather estimates if race hasn't occurred
- Outputs sorted predicted race order

🙌 Acknowledgements

Huge thanks to:
	•	FastF1 for open telemetry access
	•	XGBoost for powerful regression modeling
	•	Red Bull Racing for always keeping things interesting 😉

 

