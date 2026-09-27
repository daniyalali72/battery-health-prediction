# Predictive Maintenance for Lithium-Ion Batteries: State of Health (SoH) Estimation

## Overview
This project develops a machine learning pipeline to estimate the State of Health (SoH) and predict the degradation of lithium-ion batteries. Using the **NASA Ames Prognostics Data Repository**, the project compares a traditional machine learning baseline (XGBoost) against a deep learning sequence model (LSTM) to evaluate time-series degradation over a battery's lifecycle.

## The Dataset
The data consists of commercial 18650 lithium-ion cells run through continuous charge/discharge cycles at room temperature. 
Training Fleet: Batteries `B0005` and `B0018` 
Blind Test Set: Battery `B0006`

During each discharge cycle (2A load until 2.7V cutoff), continuous sensor arrays recorded voltage, current, and temperature. The target variable is the capacity fade over time, with the End of Life (EOL) threshold defined as a 30% drop from the nominal 2.0 Ah rating.

## Methodology

### 1. Feature Engineering
Raw, continuous sensor arrays from the `.mat` files were parsed and squashed into flat summary features per cycle to train the models:
* `Max_Temperature_C`: Peak heat stress during the cycle.
* `Avg_Temperature_C`: Sustained operating temperature.
* `Min_Voltage_V`: Lowest voltage drop before cutoff.
* `Discharge_Time_s`: Total duration of the discharge session.

### 2. Baseline Model: XGBoost Regressor
The initial baseline utilized an XGBoost Regressor. The model was trained on the engineered features of `B0005` and `B0018` and tested on the completely unseen `B0006` battery.
* **Result:** Achieved an RMSE of `0.0410`.
* **Analysis:** While highly accurate on the macro-trend, XGBoost evaluates each cycle in isolation. It lacks the mathematical memory to understand chronological wear-and-tear, resulting in slight under-predictions at the extreme end of the battery's life.

![XGBoost Results](images/xgboost_results.png)

### 3. Deep Learning Model: LSTM Neural Network
To capture the chronological nature of battery degradation, the 2D feature matrix was scaled using `MinMaxScaler` and reshaped into a 3D matrix `(Samples, Time Steps, Features)` using a 10-cycle sliding window.
* **Architecture:** 1 LSTM layer (50 units, ReLU), 20% Dropout to prevent overfitting, and a Dense output layer.
* **Result:** Achieved an RMSE of `0.0429` on the unseen `B0006` battery.

![LSTM Results](images/lstm_results.png)

## Key Insights & Limitations
* **Smoothing the Noise:** The lithium-ion cells exhibit jagged "micro-spikes" in capacity due to temporary chemical regeneration during resting periods. Because the LSTM utilizes a 10-cycle historical window, it learned to filter out this daily noise and draw a highly confident macro-trend prediction curve.
* **End-of-Life Divergence:** The LSTM slightly under-predicted the severity of the final capacity drop. Future iterations of this pipeline will experiment with expanding the sliding window to 20+ cycles to capture deeper historical momentum, or utilizing Bidirectional LSTMs to better map the accelerated degradation near the EOL threshold.

## How to Run
1. Clone the repository.
2. Install dependencies via `pip install -r requirements.txt`.
3. Download the `B0005.mat`, `B0006.mat`, and `B0018.mat` files from the [NASA Ames Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository) and place them in the `/data` directory.
4. Run the notebooks in sequential order.
