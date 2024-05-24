# Random Forest Regression: An Introduction

This repository contains a Jupyter Notebook that provides a comprehensive introduction to building and evaluating random forest regression models. The notebook demonstrates the steps involved in preprocessing data, training a random forest model, evaluating its performance, and tuning its hyperparameters for optimal results.

## Overview

### Random Forest Regression
Random forests are an ensemble learning method that constructs multiple decision trees during training and outputs the average prediction of the individual trees. This notebook covers:
- Data preprocessing
- Building and training a random forest model
- Model evaluation
- Hyperparameter tuning

## Notebook Content

### 1. Import Libraries and Data
First, we import the necessary libraries and load the dataset. We visualize the data to understand its distribution.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/house_price_by_area.csv")
X = df["LotArea"] # Independent variable 
y = df["SalePrice"] # Dependent variable 

plt.scatter(X, y)
plt.title("House Price vs Area")
plt.xlabel("Lot Area")
plt.ylabel("Sale Price")
plt.show()
```

### 2. Preprocessing
We standardize the features to ensure that the model performs optimally.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(X)[:, np.newaxis])

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=6)
```

### 3. Training
We build and train a random forest regression model with 100 trees and a maximum depth of 5.

```python
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=100, max_depth=5)
RF.fit(x_train, y_train)
```

### 4. Testing
We evaluate the model's performance by predicting on the test set and calculating the Root Mean Squared Error (RMSE).

```python
y_pred = RF.predict(x_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

f, ax = plt.subplots(figsize=(5, 5))
ax.set_title('Actual vs Predicted')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.scatter(y_test, y_pred)
ax.plot(y_test, y_test, 'r')
plt.show()
```

### 5. Tuning Model Hyperparameters
We tune the hyperparameters of the random forest model to achieve better performance. We train multiple models with different numbers of trees and compare their performance.

```python
forest_1 = RandomForestRegressor(n_estimators=2, max_depth=5, random_state=23)
forest_2 = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=23)
forest_3 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=23)

forest_1.fit(x_train, y_train)
forest_2.fit(x_train, y_train)
forest_3.fit(x_train, y_train)
```

### 6. Model Evaluation
We evaluate the performance of the different models and visualize the results.

```python
f, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=3, sharey=True)

pred = [forest_1.predict(x_test), forest_2.predict(x_test), forest_3.predict(x_test)]
title = ['trees = 2', 'trees = 20', 'trees = 100']

for i in range(3):
    rmse = round(np.sqrt(mean_squared_error(pred[i], y_test)))
    ax[i].set_title(title[i] + " (RMSE: " + str(rmse) + ")")
    ax[i].set_xlabel('Actual')
    ax[i].set_ylabel('Predicted')
    ax[i].plot(y_test, y_test, 'r')
    ax[i].scatter(y_test, pred[i])
```

### Results
The notebook demonstrates the effectiveness of random forest regression and highlights the importance of tuning hyperparameters to achieve the best model performance.

## Usage
To run this notebook, clone this repository and open the notebook in Jupyter:

```bash
git clone https://github.com/yourusername/Random-Forest-Regression.git
cd Random-Forest-Regression
jupyter notebook
```

## Conclusion
This notebook provides a step-by-step guide to building, evaluating, and tuning random forest regression models. It serves as a valuable resource for data scientists and machine learning practitioners interested in leveraging random forests for regression tasks.

Contributions and feedback are welcome! Feel free to open issues or submit pull requests to improve this repository.
