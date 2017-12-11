import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

df = pd.read_csv('boston.csv')
X = df.drop('MEDV', axis = 1).values
y = df['MEDV'].values

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size = 0.3, random_state = 42)

#using linear regression
print("#####using liner regression#####")
reg = LinearRegression()
reg.fit(X_train,y_train)

df_dummy = pd.read_csv('testdata.csv',header=None)
X_dummy = df_dummy.iloc[:,0:13].values
y_dummy = df_dummy.iloc[:,13:14].values
y_pred_dummy = reg.predict(X_dummy)
print("the predicted median value of house using linear regression is : "+str(y_pred_dummy))
y_pred = reg.predict(X_test)
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#using lasso regression
print("#####using lasso regression#####")
reg = Lasso()
reg.fit(X_train,y_train)
X_dummy = df_dummy.iloc[:,0:13].values
y_dummy = df_dummy.iloc[:,13:14].values
y_pred_dummy = reg.predict(X_dummy)
print("the predicted median value of house using lasso regression is : "+str(y_pred_dummy))
y_pred = reg.predict(X_test)
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#using ridge regression
print("#####using ridge regression#####")
reg = Ridge()
reg.fit(X_train,y_train)

df_dummy = pd.read_csv('testdata.csv',header=None)
X_dummy = df_dummy.iloc[:,0:13].values
y_dummy = df_dummy.iloc[:,13:14].values
y_pred_dummy = reg.predict(X_dummy)
print("the predicted median value of house using ridge regression is : "+str(y_pred_dummy))
y_pred = reg.predict(X_test)
print("R^2: {}".format(reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
