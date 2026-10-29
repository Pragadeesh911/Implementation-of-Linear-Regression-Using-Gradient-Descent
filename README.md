# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the dataset by selecting input features and target profit, then standardize the data.

2. Initialize the model parameters (theta) and add the intercept term to the input data.

3. Train the Linear Regression model using Gradient Descent by calculating predictions, errors, and updating theta.

4. Predict the profit for new startup data by scaling the input, calculating the prediction, and converting it back to the original scale. .
   

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: KARANKUMAR K
RegisterNumber: 212225040171

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def linear_regression (X1,y,learning_rate=0.1,num_iters = 1000):
    
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros((X.shape[1],1))
    
    for _ in range(num_iters):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data = pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")

X = data.iloc[:,:-1].values
X = X[:,[0,1,2]]
y = data.iloc[:, -1].values.reshape(-1, 1)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

print(X_scaled)

theta = linear_regression(X_scaled, y_scaled)

print("Theta:")
print(theta)


new_data = np.array([[165349.2, 136897.8, 471784.1]])
new_scaled = scaler_X.transform(new_data)
new_scaled = np.c_[np.ones((1, 1)), new_scaled]

prediction_scaled = new_scaled.dot(theta)

prediction = scaler_y.inverse_transform(prediction_scaled)

print("Scaled Prediction:", prediction_scaled)
print("Predicted Profit:", prediction)
*/
```

## Output:

<img width="250" height="438" alt="image" src="https://github.com/user-attachments/assets/aea585e1-01d1-46ce-aea2-054434ca4a49" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
