# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 20:01:26 2025

@author: LinJiamin
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats

# Define the columns to be read
plag_columns = ["SiO2_plag", "TiO2_plag", "Al2O3_plag", "FeOt_plag", "MgO_plag", "CaO_plag", "Na2O_plag", "K2O_plag"]
liq_columns = ["SIO2_melt", "TIO2_melt", "AL2O3_melt", "FEOT_melt", "CAO_melt", "MGO_melt", "NA2O_melt", "K2O_melt"]
target_column = 'H2O'

# Dataset Path
training_data_path = 'Data/Table S2 Augmented_dataset.xlsx'
target_data_path = 'INPUT_Lin2025.xlsx'

# Reading data from the training set
train_df = pd.read_excel(training_data_path)

# Select the desired column
selected_columns = plag_columns + liq_columns + [target_column]
training_data = train_df[selected_columns]

# Separation of characteristics and target variables
X_train = training_data[plag_columns + liq_columns]
y_train = training_data[target_column]

# Define the model
model = ExtraTreesRegressor(max_depth=10, min_samples_split=5, n_estimators=100, random_state=42)

# Reading test set data
test_df = pd.read_excel(target_data_path)

# Select the feature columns needed for the test set
test_features = test_df[plag_columns + liq_columns]

# Training the model and predicting the water content of the new dataset
model.fit(X_train, y_train)
test_df['H2O_ExtraTrees'] = model.predict(test_features)

# Export the predictions to a new Excel spreadsheet
output_path = 'Data/predicted_H2O_final_without_TP.xlsx'
test_df.to_excel(output_path, index=False)

print(f"Predictions have been saved to: {output_path}")
