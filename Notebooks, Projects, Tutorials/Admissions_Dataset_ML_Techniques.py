#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:09:29 2019

@author: ryantomiyama
"""


############ Machine Learning Techniques on Admissions Dataset############
### The goal is to use ML techniques to predict the chance of applicants getting admitted to graduate school.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set the path and read in data
path = "/Users/ryantomiyama/Desktop/Data/Admission_Predict.csv"
df = pd.read_csv(path, sep=",")

df.info()
df.head()
df.tail()
list(df)
df.shape # 400 observations and 9 features


# Rename variable
df = df.rename(columns = {'Chance of Admit ':'Chance of Admit'})

# Make a correlation matrix among features.
# From the figure CGPA and GRE Score are important for the response (Chance of Admit).
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, annot=True)


### Exploratory Data Analysis

# Research Experience
print('Not having research:',len(df[df.Research == 0])) # 181 have no research
print('Having research:',len(df[df.Research == 1])) # 219 have research
y = np.array([len(df[df.Research == 0]),len(df[df.Research == 1])])
x = ['Not having research', 'Having research']
plt.bar(x,y)
plt.title('Research Experience')
plt.ylabel('Frequency')

# GRE Score
df['GRE Score'].plot(kind='hist', bins=100)
plt.title('GRE Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

# CGPA 
# Note that the CGPA has a possible max value of 10. This is not on a 4.0 scale.
df['CGPA'].plot(kind='hist', bins=60)
plt.title('CGPA')
plt.xlabel('CGPA')
plt.ylabel('Frequency')

# Relationship between CGPA and GRE Score
# The plot shows that applicants with higher GRE Scores tend to have higher CGPA.
plt.scatter(df['GRE Score'], df['CGPA'])
plt.title('CGPA and GRE Scores')
plt.xlabel('GRE Score')
plt.ylabel('CGPA')


### Regression Algorithms

# Drop the serial number feature
serialNo = df['Serial No.'].values
df.drop(['Serial No.'], axis=1, inplace=True)

# train_test_split splits the data into random train (80%) and test (20%) subsets
y = df['Chance of Admit'].values
x = df.drop(['Chance of Admit'], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state=1)

# Normalization
from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0,1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])


### Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_test_predict = lr.predict(x_test)
print('real value of y_test[0]: ' + str(y_test[0]) + ' -> the predict: ' + str(lr.predict(x_test.iloc[[0],:])))
print('real value of y_test[1]: ' + str(y_test[1]) + ' -> the predict: ' + str(lr.predict(x_test.iloc[[1],:])))

from sklearn.metrics import r2_score
print('r_square score: ', r2_score(y_test, lr_test_predict))
lr_train_predict = lr.predict(x_train)
print('r_square score (train dataset): ', r2_score(y_train, lr_train_predict))


### Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 100, random_state =1)
rfr.fit(x_train, y_train)
rfr_test_predict = rfr.predict(x_test)
print('real value of y_test[0]: ' + str(y_test[0]) + ' -> the predict: ' + str(rfr.predict(x_test.iloc[[0],:])))
print('real value of y_test[1]: ' + str(y_test[1]) + ' -> the predict: ' + str(rfr.predict(x_test.iloc[[1],:])))


from sklearn.metrics import r2_score
print('r_square score: ', r2_score(y_test, rfr_test_predict))
rf_train_predict = rfr.predict(x_train)
print('r_square score (train dataset): ', r2_score(y_train, rf_train_predict))

### Comparing Regression Methods
y = np.array([r2_score(y_test, lr_test_predict), r2_score(y_test, rfr_test_predict)])
x = ['Linear Regression', 'Random Forest Regression']
plt.bar(x,y)
plt.title('Comparing Linear and Random Forest Regression')
plt.xlabel('Method')
plt.ylabel('r_2 Score')

#### In terms of these two regression methods they do similaly well.



### Classification Algorithms

# If the response is more than 0.8 than it is classified as a 1, if not then a 0.
y_train_1 = [1 if each > 0.8 else 0 for each in y_train]
y_test_1 = [1 if each > 0.8 else 0 for each in y_test]
y_train_1 = np.array(y_train_1)
y_test_1 = np.array(y_test_1)

### Support Vector Machine
from sklearn.svm import SVC
svm = SVC(random_state=1)
svm.fit(x_train, y_train_1)
print('score: ', svm.score(x_test, y_test_1))
print('real value of y_test_1[0]: ' + str(y_test_1[0]) + ' -> the predict: ' + str(svm.predict(x_test.iloc[[0],:])))
print('real value of y_test_1[1]: ' + str(y_test_1[1]) + ' -> the predict: ' + str(svm.predict(x_test.iloc[[1],:])))

from sklearn.metrics import confusion_matrix
# On the test dataset the misclassification rate is 6/80 or 0.075
cm_svm = confusion_matrix(y_test_1, svm.predict(x_test))
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm_svm, annot = True, linewidths=0.5, linecolor='white', fmt='.0f', ax=ax)
plt.title('Test for Test Dataset')
plt.xlabel('Predicted y Values')
plt.ylabel('Real y Values')

# On the train dataset the misclassification rate is 21/320 or 0.065625
cm_svm_train = confusion_matrix(y_train_1, svm.predict(x_train))
f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(cm_svm_train, annot = True, linewidths=0.5, linecolor='white', fmt='.0f', ax=ax)
plt.title('Test for Test Dataset')
plt.xlabel('Predicted y Values')
plt.ylabel('Real y Values')


### Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, random_state=1)
rfc.fit(x_train, y_train_1)
print('score: ', rfc.score(x_test, y_test_1))
print('real value of y_test_1[0]: ' + str(y_test_1[0]) + ' -> the predict: ' + str(rfc.predict(x_test.iloc[[0],:])))
print('real value of y_test_1[1]: ' + str(y_test_1[1]) + ' -> the predict: ' + str(rfc.predict(x_test.iloc[[1],:])))


from sklearn.metrics import confusion_matrix
# On the test dataset the misclassification rate is 6/80 or 0.075
cm_rfc = confusion_matrix(y_test_1, rfc.predict(x_test))
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_rfc, annot=True, linewidth=0.5, linecolor='white', fmt='.0f', ax=ax)
plt.title('Test for Test Dataset')
plt.xlabel('Predicted y Values')
plt.ylabel('Real y Values')

# On the train dataset the misclassification rate is 0/320 or 0
cm_rfc = confusion_matrix(y_train_1, rfc.predict(x_train))
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm_rfc, annot=True, linewidth=0.5, linecolor='white', fmt='.0f', ax=ax)
plt.title('Test for Train Dataset')
plt.xlabel('Predicted y Values')
plt.ylabel('Real y Values')

### Based on these two classifcation methods it appears that they both perform similarly well.


























