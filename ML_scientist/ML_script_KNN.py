# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:58:15 2020

@author: sachin.kalra
"""
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Loading Dataset
df=datasets.load_iris()

df.keys()
df.data.shape

df.target_names

x=df.data
y=df.target
df=pd.DataFrame(x,columns=df.feature_names)
df.head()
#Visualizae
pd.plotting.scatter_matrix(frame=df,c=y,figsize=[8,12],s=150,marker='D')
# c=color

# ALl ML models in python are implemented as Python classes
# .fit() trains ML class object and stores information

# 1 : KNN in Python

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X=df,y=y)
prediction=knn.predict(X=df.loc[:10,:])

#train-test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,
                                               random_state=21,stratify=y)

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X=x_train,y=y_train)
y_pred=knn.predict(X=x_test)

print("Prediction : \n{}".format(y_pred))

#accuracy:
#by logic:
print("Accuracy : {}" .format(round(100*(y_pred==y_test).sum()/len(y_test)),2))
#by function
knn.score(X=x_test,y=y_test )

#Model Complexity Curve
#very high k = underfitting
#low k = overfitting

#running KNN in Loop:

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X=x_train,y=y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(x_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# Performace Metrics


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

df=datasets.load_iris()

x=df.data
y=df.target

#train-test split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.4,random_state=42)
#knn modelling
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train,y_train)
#prediction
y_pred=knn.predict(x_test)

print(confusion_matrix(y_test, y_pred,))
print(classification_report(y_test, y_pred))

#ROC Curve

from sklearn.metrics import roc_curve

y_pred_prob=knn.predict_proba(x_test)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
#doesn't work on multiclass problem