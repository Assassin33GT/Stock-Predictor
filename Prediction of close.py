# -*- coding: utf-8 -*-
"""
Created on Mon May 20 01:30:39 2024

@author: user
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load data
data = pd.read_csv("C:/Users/user/Desktop/AAPL.csv")

# Drop unnecessary columns
data = data.drop(['Date', 'Volume'], axis=1)

# Create lagged features for closing prices
data['Close_lag1'] = data['Close'].shift(1)
data['Close_lag2'] = data['Close'].shift(2)
data['Close_lag3'] = data['Close'].shift(3)

# Drop rows with NaN values
data = data.dropna()

# Define features (X) and target (Y)
X = data[['Close_lag1', 'Close_lag2', 'Close_lag3']]
Y = data['Close']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# Normalize the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=101)
regressor.fit(X_train_scaled, Y_train)

# Predict
Y_pred = regressor.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(Y_test, Y_pred)
print("R-squared:", r2)

# Predict future closing prices based on the last available data
last_known = X.iloc[-1, :].values.reshape(1, -1)
last_known_scaled = scaler.transform(last_known)

predicted_values = []

for i in range(1, 11):
    next_close = regressor.predict(last_known_scaled)
    predicted_values.append(next_close[0])
    # Update the last_known_scaled with the new prediction
    last_known = np.roll(last_known, shift=-1)
    last_known[0, -1] = next_close
    last_known_scaled = scaler.transform(last_known)

print("Predicted next ten closing prices aft:")
for value in predicted_values:
    print(value)
