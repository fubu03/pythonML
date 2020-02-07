# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:05:33 2020

@author: sachin.kalra
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import  matplotlib.pyplot as plt

logreg=LogisticRegression()
x_train,y_train,x_test,y_test=train_test_split(X,y,test_size=0.4,random_state=42)
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#ROC Curve
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# outouts two columns for class probabilities
# these probabilities are for class in order labels are defined
# 1 is used for checking probabilities for class under observation
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

#AUC Curve

from sklearn.metrics import roc_auc_score

logreg=LogisticRegression()

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

logreg.fit(x_train,y_train)
y_pred_prob=logreg.predict_proba(x_test)[:,1]
roc_auc_score(y_test, y_pred_prob)

#auc by cross validation

from sklearn.model_selection import cross_val_score
cv_scores=cross_val_score(logreg,X,y,cv=5,scoring='roc_auc')



