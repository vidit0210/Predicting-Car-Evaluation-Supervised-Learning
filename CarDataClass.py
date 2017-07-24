from sklearn import cross_validation,neighbors,svm
import pandas as pd
import numpy as np

'''
Developed By Vidit Shah
Data Set source UCI
'''

#Read the data
df = pd.read_csv("car.data.txt")

'''
We need to numbers in numpy array thus we replace String Arguments with numbers with dictionary
'''
df.replace({'vhigh': 99, 'high': 88, 'med': 77, 'low': 66,'small':55,
            'vgood':1,'good':2,'acc':3,'unacc':4,
            'more':5,'5more':6,'big':7}, inplace=True)

print(df.head())
X= np.array(df.drop(['label'],1))
y= np.array(df['label'])

#Splitting into training and testing data sets
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)


#Using KNeighbor Algorithm
clf= neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)

case1= np.array([99,99,2,2,55,66])
pred=clf.predict(case1)
print(pred)

#Using Support Vector Machine Algorithm
clf= svm.SVC()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)

case1= np.array([99,99,2,2,55,66])
pred=clf.predict(case1)
print(pred)