# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:45:11 2020

@author: sachin.kalra
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
boston=pd.read_csv(r'C:\Users\sachin.kalra\Desktop\nlp_ss\ML_scientist/boston.csv')

boston.columns
boston.shape
boston.isnull().sum().sum()

#target var = MEDV

X=boston.drop('medv',axis=1).values
y=boston['medv'].values



#predicting price from single feature rooms
x_rooms=X[:,5]

y=y.reshape(-1,1)
x_rooms = x_rooms.reshape(-1,1)

x = np.array([[2,3,4], [5,6,7]])

np.reshape(x, (-2, 2))

plt.scatter(x_rooms,y)
plt.xlabel('number of rooms')
plt.ylabel('value of house /1000 in $')
plt.show()

#more rooms = higher prices
#quick Regression

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(X=x_rooms,y=y)
#quick visual
prediction_space=np.linspace(min(x_rooms),max(x_rooms)).reshape(-1,1)
plt.scatter(x_rooms,y,color='blue')
plt.plot(prediction_space,reg.predict(prediction_space),color='red',linewidth=4)
plt.show()

#heatmap of num columns
tmp=boston.select_dtypes(include='float64')
sns.heatmap(boston.corr(),square=True, cmap='RdYlGn')

#ML-Regression

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X=boston.drop('medv',axis=1)
y=boston['medv']
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

reg_all=LinearRegression()

reg_all.fit(x_train,y_train)
y_pred=reg_all.predict(x_test)
#R squared default
reg_all.score(x_test,y_test)

plt.plot(x_test.age,y_pred,color='red',linewidth=2)
plt.show()

#RMSE

from sklearn.metrics import mean_squared_error

print("R^2: {}".format(reg_all.score(x_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#Cross Validation
from sklearn.model_selection import cross_val_score

reg=LinearRegression()
cv_results=cross_val_score(reg,X,y,cv=5)
np.mean(cv_results)
#rsquared is default 

# L1 L2 regression

#ridge L1
from sklearn.linear_model import Ridge

ridge= Ridge(alpha=0.0001,normalize=True)
#normalize = True scales all variables
ridge.fit(x_train,y_train)
ridge_pred=ridge.predict(x_test)

print("R^2: {}".format(ridge.score(x_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,ridge_pred))
print("Root Mean Squared Error: {}".format(rmse))

#Lasso L2
from sklearn.linear_model import Lasso

lasso= Lasso(alpha=0.0001,normalize=True)
#normalize = True scales all variables
lasso.fit(x_train,y_train)
lasso_pred=lasso.predict(x_test)

print("R^2: {}".format(lasso.score(x_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test,lasso_pred))
print("Root Mean Squared Error: {}".format(rmse))

#important features in Lasso

names=boston.drop('medv',axis=1).columns
lasso=Lasso(alpha=0.1)
lasso_coef=lasso.fit(X,y).coef_

plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names,rotation=60)
plt.ylabel('Coefficients')
plt.show()

#checking model with lowest alpha and K fold + For Loop

#alpha space
alpha_space= np.logspace(-4,0,50)
ridge_scores=[]
ridge_scores_std=[]

ridge=Ridge(normalize=True)


#groundbreaking display function dcmp

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

for alpha in alpha_space:
    ridge.alpha=alpha
    ridge_cv_scores=cross_val_score(ridge,X,y,cv=10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.mean(ridge_cv_scores))
    print('alpha : {}'.format(alpha))
    print('ridge_scores : {}'.format(np.mean(ridge_cv_scores)))

display_plot(ridge_scores,ridge_scores_std)
