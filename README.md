# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 



### AIM:
To Implementat an Auto Regressive Model using Python for daily website visitors dataset.
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

file_path = '/content/daily_website_visitors.csv'
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data['Page.Loads'] = pd.to_numeric(data['Page.Loads'].str.replace(',', ''), errors='coerce')
data.dropna(subset=['Page.Loads'], inplace=True)

page_loads = data['Page.Loads']

diff_data = page_loads.diff().dropna()

result = adfuller(diff_data)
print('ADF Statistic (After Differencing):', result[0])
print('p-value (After Differencing):', result[1])

train_data = diff_data[:int(0.8 * len(diff_data))]
test_data = diff_data[int(0.8 * len(diff_data)):]

lag_order = 13
model = AutoReg(train_data, lags=lag_order)
model_fit = model.fit()

max_lags = len(diff_data) // 2

plt.figure(figsize=(10, 6))
plot_acf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Differenced Page Loads')
plt.show()

plt.figure(figsize=(10, 6))
plot_pacf(diff_data, lags=max_lags, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Differenced Page Loads')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

mse = mean_squared_error(test_data, predictions)
print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data.values, label='Test Data - Differenced Page Loads', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Differenced Page Loads', color='orange', linestyle='--', linewidth=2)
plt.xlabel('DateTime')
plt.ylabel('Differenced Page Loads')
plt.title('AR Model Predictions vs Test Data (Differenced Page Loads)')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

#### GIVEN DATA
![image](https://github.com/user-attachments/assets/6c135ca8-6ae8-4d6b-abd8-136145b6f886)

#### ADFuller
```
ADF Statistic (After Differencing): -12.250174139901453
p-value (After Differencing): 9.580241999169595e-23
```

#### PACF - ACF
![image](https://github.com/user-attachments/assets/434d133c-2d19-4ad1-86d8-05296c3bd15f)

### MSE
```
Mean Squared Error (MSE): 333851.7043400734
```
#### PREDICTION
![image](https://github.com/user-attachments/assets/00f4f9f5-a019-4712-801d-8a7921d1a812)


### RESULT:
Thus we have successfully implemented the auto regression function using python for daily website visitors dataset.
