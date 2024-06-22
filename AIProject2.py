import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np


data1=pd.read_csv("C:/Users/user/Desktop/AAPL.csv")
data1.info() #no. of rows and columns



data1=data1.drop('Date',axis=1)
data1=data1.drop('Volume',axis=1)
X1 = data1.iloc[:,data1.columns!='Close'].values
#X1 = np.array(data1.index).reshape(-1,1)
Y1 = data1['Close']
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.2, random_state=101)


#normalize data
scaler1 = StandardScaler().fit(X_train1)
X_train_scaled1 = scaler1.transform(X_train1)
X_test_scaled1 = scaler1.transform(X_test1)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train_scaled1, Y_train1)
Y_pred1 = lm.predict(X_test_scaled1)

r21 = r2_score(Y_test1, Y_pred1)
print("R-squared:", r21)

#Cross Validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(lm, X_train_scaled1, Y_train1, cv=5, scoring='r2')
print("Cross-validated R-squared scores:", cv_scores)
print("Mean cross-validated R-squared:", cv_scores.mean())

plt.figure(figsize=(10, 6))
plt.scatter(Y_test1, Y_pred1, color='blue', alpha=0.5)
plt.plot([Y_test1.min(), Y_test1.max()], [Y_test1.min(), Y_test1.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Stock Prices')
plt.show()



#Regression Tree
data=pd.read_csv("C:/Users/user/Desktop/AAPL.csv")

data=data.drop('Date',axis=1)
data=data.drop('Volume',axis=1)
X = data.drop(['Close'], axis=1).values
Y = data['Close']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

# Normalize data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=101)
regressor.fit(X_train_scaled, Y_train)

# Predict
Y_pred = regressor.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(Y_test, Y_pred)
print("R-squared:", r2)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)  # Diagonal line
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Stock Prices')
plt.show()



# Perform cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(regressor, X_train_scaled, Y_train, cv=5, scoring='r2')
print("Cross-validated R-squared scores:", cv_scores)
print("Mean cross-validated R-squared:", cv_scores.mean())
