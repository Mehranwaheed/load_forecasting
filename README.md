# Project Overview
In the dynamic landscape of energy management, the project focuses on load forecasting for the city of Tétouan in Morocco. The goal is to predict electricity demand, enabling better resource allocation, and contributing to a more efficient and sustainable energy infrastructure. This research endeavors to construct and assess various machine learning models for load forecasting, with the objective of comparing their outcomes to identify the most effective algorithm. Additionally, the study will delve into enhancing the performance of these models and pinpointing the most influential predictors in the forecasting process.

# Data Preprocessing
imported necessary libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
```
dataset loading into pandas dataframe

```python
#data reading
df=pd.read_csv("/content/drive/MyDrive/Datasets/Tetuan City power consumption.csv")
```
first look of the data
```python
df.head()
```
| DateTime        | Temperature | Humidity | Wind Speed | general diffuse flows | diffuse flows | Zone 1 Power Consumption | Zone 2 Power Consumption | Zone 3 Power Consumption |
|-----------------|-------------|----------|------------|------------------------|---------------|---------------------------|---------------------------|---------------------------|
| 1/1/2017 0:00   | 6.559       | 73.8     | 0.083      | 0.051                  | 0.119         | 34055.69620               | 16128.87538               | 20240.96386               |
| 1/1/2017 0:10   | 6.414       | 74.5     | 0.083      | 0.070                  | 0.085         | 29814.68354               | 19375.07599               | 20131.08434               |
| 1/1/2017 0:20   | 6.313       | 74.5     | 0.080      | 0.062                  | 0.100         | 29128.10127               | 19006.68693               | 19668.43373               |
| 1/1/2017 0:30   | 6.121       | 75.0     | 0.083      | 0.091                  | 0.096         | 28228.86076               | 18361.09422               | 18899.27711               |
| 1/1/2017 0:40   | 5.921       | 75.7     | 0.081      | 0.048                  | 0.085         | 27335.69620               | 17872.34043               | 18442.40964               |


There are 52416 rows of data, each containing 9 distinct features.
```python
print(f"The dataset have {df.shape[1]} rows and {df.shape[0]} features")
```

## Missing Values
Fortunately, the dataset did not contain any forms of missing values.
```python
df.isna().sum()
```
|column Name| Missing Values|
|------------- |:-------------:|
|DateTime|                     0
|Temperature|                  0
|Humidity|                     0
|Wind Speed|                   0
|general diffuse flows|        0
|diffuse flows|                0
|Zone 1 Power Consumption|     0
|Zone 2  Power Consumption |   0
|Zone 3  Power Consumption|    0

## Data type
```python
df.info()
```
|   Column                    | Non-Null Count | Dtype   |
|-----------------------------|----------------|---------|
| DateTime                    | 52416 non-null | object  |
| Temperature                 | 52416 non-null | float64 |
| Humidity                    | 52416 non-null | float64 |
| Wind Speed                  | 52416 non-null | float64 |
| general diffuse flows       | 52416 non-null | float64 |
| diffuse flows               | 52416 non-null | float64 |
| Zone 1 Power Consumption    | 52416 non-null | float64 |
| Zone 2 Power Consumption    | 52416 non-null | float64 |
| Zone 3 Power Consumption    | 52416 non-null | float64 |

It was noticed that while all other features had the accurate data type of float, the DateTime feature exhibited an object data type.


The DateTime feature was consequently transformed into the datetime format using the Pandas datetime function, rendering it suitable for further analysis.
```python
df['DateTime']=pd.to_datetime(df['DateTime'])

```
|   Column                    | Non-Null Count | Dtype   |
|-----------------------------|----------------|---------|
| DateTime                    | 52416 non-null | datetime64[ns]  |
| Temperature                 | 52416 non-null | float64 |
| Humidity                    | 52416 non-null | float64 |
| Wind Speed                  | 52416 non-null | float64 |
| general diffuse flows       | 52416 non-null | float64 |
| diffuse flows               | 52416 non-null | float64 |
| Zone 1 Power Consumption    | 52416 non-null | float64 |
| Zone 2 Power Consumption    | 52416 non-null | float64 |
| Zone 3 Power Consumption    | 52416 non-null | float64 |


To facilitate analysis and gain insights from the data, details such as the day, month, quarter of the year, day of the week, day of the year, day of the month, and week of the year were extracted.
```python

df['day'] = df['DateTime'].dt.day
df['day_of_week'] = df['DateTime'].dt.dayofweek  # Monday is 0 and Sunday is 6
df['day_of_year'] = df['DateTime'].dt.dayofyear
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute
df['month'] = df['DateTime'].dt.month
df['quarter'] = df['DateTime'].dt.quarter
df['week'] = df['DateTime'].dt.isocalendar().week
df['week_of_year'] = df['DateTime'].dt.week
```
The DateTime column was set as the index to facilitate data analyses.
``` python
df.set_index("DateTime",inplace=True)
```
renamed the column names such as first Zone is Quads, Zone one is Boussafu and Smir last one.

```pyhton

df.rename(columns={"Zone 1 Power Consumption":"Quads",'Zone 2  Power Consumption':"Boussafu",'Zone 3  Power Consumption':"Smir"},inplace=True)

```
# Outliers
```python

sns.boxplot(df['Quads'])
plt.ylabel("KWh")
plt.title("Quads Electricity consumption")
plt.show()

```
![quads](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/a45db34e-ce5b-4a19-a743-16fb8ccafdea)

there are no outliers concerning the load consumption for Quads region

```python

sns.boxplot(df['Smir'])
plt.ylabel("KWh")
plt.title("Smir Electricity consumption")
plt.show()
```
```python
sns.boxplot(df['Boussafu'])
plt.ylabel("KWh")
plt.title("Boussafu Electricity consumption")
plt.show()
```
![smir](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/70d55c88-ac0a-43ef-8779-96522c408ba0) 
![bossafu](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/5c373d69-2d81-4bb9-9213-251e30687b07)

from the boxplot, it is observed that in both the Smir and Boussafu Region a data points from 35,000 kWh and above are projected as outliers and thus need to be removed from the dataset.

removed the outliers from the dataset
``` python
df = df[(df['Smir'] <= 35000) & (df['Boussafu'] <= 35000)]
```
# Stationarity

```python

time_series_data = df['Quads']


result = adfuller(time_series_data)


print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

if result[1] <= 0.05:
    print("Reject the null hypothesis - The data is stationary.")
else:
    print("Fail to reject the null hypothesis - The data is non-stationary.")

```
**ADF Test Results:**

- **ADF Statistic:** -37.060825384600506
- **p-value:** 0.0
- **Critical Values:**
  - 1%: -3.430477550221914
  - 5%: -2.861596374609351
  - 10%: -2.5668000063347174

**Conclusion:**
The ADF test was conducted to assess the stationarity of the time series data. The results indicate that the ADF Statistic is significantly below the critical values, with a p-value of 0.0. Therefore, we reject the null hypothesis, suggesting that the data is stationary.

# Normalization

As data is collected from different resources, so it's better to do the normalization. All the columns except the date time columns are normalized using the Minmax scaler technique to transform the values of numeric columns to a common scale, without distorting differences in the range of the values by converting the mean of the observed values to 0 and the standard deviation to 1.

![Screenshot 2024-01-24 130540](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/416a2a0c-dfe5-4378-ae19-2089518c6134)

```python
df_normalized=df.copy()
datetime_columns = ['day_of_year',"hour",'minute', 'month', 'quarter', 'week','week_of_year','day', 'day_of_week',]
df_datetime = df_normalized[datetime_columns]
df_normalized = df_normalized.drop(columns=datetime_columns)

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the remaining columns
df_normalized[df_normalized.columns] = scaler.fit_transform(df_normalized[df_normalized.columns])

# Add the DateTime columns back to the normalized DataFrame
df_normalized = pd.concat([df_datetime, df_normalized], axis=1)

# Display the normalized DataFrame
df_normalized.head()
```
| DateTime               | day_of_year | hour | minute | month | quarter | week | week_of_year | day | day_of_week | Temperature | Humidity | Wind Speed | general diffuse flows | diffuse flows | Quads | Boussafu | Smir     |
|------------------------|-------------|------|--------|-------|---------|------|--------------|-----|-------------|-------------|----------|------------|------------------------|---------------|-------|----------|----------|
| 2017-01-01 00:00:00   | 1           | 0    | 0      | 1     | 1       | 52   | 52           | 1   | 6           | 0.090091    | 0.748382 | 0.005130   | 0.000040               | 0.000115      | 0.561382 | 0.286320 | 0.492206 |
| 2017-01-01 00:10:00   | 1           | 0    | 10     | 1     | 1       | 52   | 52           | 1   | 6           | 0.086146    | 0.756770 | 0.005130   | 0.000057               | 0.000079      | 0.443286 | 0.409121 | 0.488425 |
| 2017-01-01 00:20:00   | 1           | 0    | 20     | 1     | 1       | 52   | 52           | 1   | 6           | 0.083399    | 0.756770 | 0.004663   | 0.000050               | 0.000095      | 0.424167 | 0.395185 | 0.472507 |
| 2017-01-01 00:30:00   | 1           | 0    | 30     | 1     | 1       | 52   | 52           | 1   | 6           | 0.078176    | 0.762761 | 0.005130   | 0.000075               | 0.000091      | 0.399126 | 0.370763 | 0.446044 |
| 2017-01-01 00:40:00   | 1           | 0    | 40     | 1     | 1       | 52   | 52           | 1   | 6           | 0.072736    | 0.771148 | 0.004819   | 0.000038               | 0.000079      | 0.374255 | 0.352274 | 0.430325 |


# Multicollinearity

To check Multicollinearity used the Pearson’s correlation coefficient. The independent variables for this study are hour of the day, month in the year, quarter of the year, day of the week, day of the year, day of the month and week of year, Temperature, Humidity, Wind Speed, general diffuse flows and diffuse flows while the dependent variable are the load consumptions for the various regions represented by Quads, Smir and Boussafou.

```python
correlation_matrix=df_normalized.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
```
![cor](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/0bf83c4a-0693-4658-b0aa-be9b24f75bf7)

as observed from the above heatmap there's multicollinearity exists between diffuse flows and general diffuse flow so have to remove one of them. similarly multicollinearity exists between the regions are 70% or more so for this project, dropping diffuse flow , Boussafu, and Smir regions and focusing on just Quads Region.
the multicollinearity exists between other variables as well but that is not significant.

```python
df_normalized.drop(columns=["diffuse flows","Boussafu","Smir"],inplace=True)
df_normalized.head()
```
| DateTime               | day_of_year | hour | minute | month | quarter | week | week_of_year | day | day_of_week | Temperature | Humidity | Wind Speed | general diffuse flows | Quads      |
|------------------------|-------------|------|--------|-------|---------|------|--------------|-----|-------------|-------------|----------|------------|------------------------|------------|
| 2017-01-01 00:00:00   | 1           | 0    | 0      | 1     | 1       | 52   | 52           | 1   | 6           | 0.090091    | 0.748382 | 0.005130   | 0.000040               | 0.561382   |
| 2017-01-01 00:10:00   | 1           | 0    | 10     | 1     | 1       | 52   | 52           | 1   | 6           | 0.086146    | 0.756770 | 0.005130   | 0.000057               | 0.443286   |
| 2017-01-01 00:20:00   | 1           | 0    | 20     | 1     | 1       | 52   | 52           | 1   | 6           | 0.083399    | 0.756770 | 0.004663   | 0.000050               | 0.424167   |
| 2017-01-01 00:30:00   | 1           | 0    | 30     | 1     | 1       | 52   | 52           | 1   | 6           | 0.078176    | 0.762761 | 0.005130   | 0.000075               | 0.399126   |
| 2017-01-01 00:40:00   | 1           | 0    | 40     | 1     | 1       | 52   | 52           | 1   | 6           | 0.072736    | 0.771148 | 0.004819   | 0.000038               | 0.374255   |

# Exploratory Data Analysis

```python
plt.figure(figsize=(28,6))
plt.plot(df_normalized.index, df_normalized['Quads'])
plt.xlabel("Date")
plt.ylabel("Load Consumption Normalized")
plt.title("Load Consumption of Quads Region for 2017")
plt.show()
```
![load](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/2fc3dc80-20da-46d5-ae64-7637f40c1b53)

taking mean of load for the month to see the clear pattern of the load consumption per month
``` python
mean_by_month = df.resample('M').mean()
plt.figure(figsize=(20,6))
sns.barplot(x='DateTime', y='Quads', data=mean_by_month)
plt.title("load Consumption per month")
plt.show()
```
![download (6)](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/643f7340-f81f-4390-a370-f73401cc3dfa)

The Electricity consumption is stable form january till april then reaches 35000 kwh for next 5,6 months and then start decreasing.

furthermore explored the hourly load consumption
```python
plt.figure(figsize=(10, 6))
sns.set_style('whitegrid')
sns.boxplot(x='hour', y='Quads', data=df_normalized)
plt.title('Hourly load consumption')

plt.show()

```
![hourly](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/5f37700f-5004-4fea-808a-05ce41788f4a)

hour of the day load consumption with 08:00 PM showing the highest load consumption units and 08:00 PM also has the highest median load consumption. The hour of the day with lowest load consumption unit is 04:00 AM.

# Machine learning Modelling
After completing the data preprocessing and exploratory analysis, gaining valuable insights into the dataset, we are ready to progress to the next step of modeling. To proceed, we need to partition the data into training and testing sets. For this purpose, we have determined that 70% of the data will be utilized for training the model, while the remaining 30% will be reserved for evaluating the model's performance.
```python
x=round(df_normalized.shape[0]*0.70)
X=df_normalized.drop(columns=["Quads"])
y=df_normalized["Quads"]
X_train=X.iloc[0:x]
X_test=X.iloc[x:]
y_train=y.iloc[:x]
y_test=y.iloc[x:]
```
## Feature Importance
To assess the significance of features in our analysis, we opted to employ the Random Forest algorithm, acknowledging that various other techniques are also available for this purpose.

``` python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot feature importance
sns.set_style('white')
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
```
``` python
feature_importance.plot(kind='bar')
plt.title('Random Forest Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=60)
#plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
plt.show()
```
![fe](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/09ae74f9-82d2-47d9-9411-92a6c946a750)

The features of "hour" and "temperature" have been identified as the most crucial factors in the analysis.

# Random Forest Regressor

Optimizing the performance of the Random Forest algorithm involves tuning several hyperparameters. Selecting the most suitable values for these parameters is a critical step to achieve accurate results. Employing grid search cross-validation (Grid Search CV) was utilized to systematically explore and identify the best set of hyperparameters, considering there is no fixed rule for their determination.

```python

rf_model = RandomForestRegressor()

# Define the parameter grid for the grid search
param_grid = {
    'n_estimators': [20, 30],
    'max_features': [1,  7], #['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10],
    'min_samples_split': [2, 3],
    'min_samples_leaf': [1, 2]
}

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding score
print("Best Parameters: ", grid_search.best_params_)
print("Best Mean Squared Error: ", -grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_

```
**Best Parameters:**
- `max_depth`: None
- `max_features`: 7
- `min_samples_leaf`: 2
- `min_samples_split`: 3
- `n_estimators`: 20

**Best Mean Squared Error:**
0.05030643422165748

```python
best_model.fit(X_train,y_train)

y_pred_test_rf = best_model.predict(X_test)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred_test_rf)

print("Mean Absolute Error on Test Set: ", mae)

mse = mean_squared_error(y_test, y_pred_test_rf)

# Calculate root mean squared error
rmse = np.sqrt(mse)

print("Root Mean Squared Error on Test Set:", rmse)
```
**Mean Absolute Error on Test Set:** 0.10447929319190494

**Root Mean Squared Error on Test Set:** 0.1239078151763514

# Decision Tree
```python
param_grid = {
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4,10],
    'max_features': [1,2,3,6,9]
}

# Decision tree model
dt_model = DecisionTreeRegressor()

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding score
print("Best Parameters: ", grid_search.best_params_)
print("Best Mean Squared Error: ", -grid_search.best_score_)
best_model = grid_search.best_estimator_
```
**Best Parameters:**
- `max_depth`: None
- `max_features`: 9
- `min_samples_leaf`: 10
- `min_samples_split`: 5

**Best Mean Squared Error:**
0.0063808723157617025

```python
from sklearn.metrics import mean_absolute_error

best_model.fit(X_train,y_train)
# Make predictions on the test set
y_pred_test_dt = best_model.predict(X_test)

# Calculate mean absolute error
mae = mean_absolute_error(y_test, y_pred_test_dt)

print("Mean Absolute Error on Test Set: ", mae)

mse = mean_squared_error(y_test, y_pred_test_dt)

# Calculate root mean squared error
rmse = np.sqrt(mse)

print("Root Mean Squared Error on Test Set:", rmse)
```
**Mean Absolute Error on Test Set:** 0.10102824130818078

**Root Mean Squared Error on Test Set:** 0.1206167295539556
# Feed Forward Neural Network

```python
X_train=tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test=tf.convert_to_tensor(X_test, dtype=tf.float32)



model = keras.Sequential([
    keras.layers.Dense(10, activation='selu', kernel_initializer='glorot_uniform', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(1)  
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with SGD, 0.9 momentum, and 100 epochs
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=2)
```

```python
y_pred_test_ffn = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred_test_ffn)

print("Mean Absolute Error on test Set: ", mae)

# Calculate Mean Squared Error (MSE) on the test set
mse = mean_squared_error(y_test, y_pred_test_ffn)
rmse = np.sqrt(mse)
print("Mean Squared Error on test Set:", rmse)
```
**Mean Absolute Error on test Set:**  0.2842636537903618

**Mean Squared Error on test Set:** 0.3074987880574227
# Results

The **Decision Tree** is considered the best among the models based on the results due to its performance metrics. The key indicators, Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), are crucial in evaluating the accuracy of regression models. In this context, the **Decision Tree** achieved the lowest MAE of **0.10**  and the lowest RMSE of **0.12**, which are favorable outcomes. Lower values for these metrics indicate that the model's predictions are closer to the actual values, reflecting a higher level of accuracy. Therefore,**Decision Tree**, with its superior performance in minimizing errors, stands out as the most effective model among the alternatives considered.

| Model                        | Mean Absolute Error on Test Set | Root Mean Squared Error on Test Set |
|------------------------------|---------------------------------|--------------------------------------|
| Random Forest                | 0.10447929319190494             | 0.1239078151763514                   |
| **Decision Tree**                | **0.10102824130818078**             | **0.1206167295539556**                   |
| Feed Forward Neural Network  | 0.2842636537903618          | 0.3074987880574227               |

![download (7)](https://github.com/Mehranwaheed/load_forecasting/assets/119947085/ddd2a465-9ec0-4c43-8611-73827a59bf23)

