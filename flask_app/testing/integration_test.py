'''
For integration testing, I have run individual cases from the test file in the both the app and through the model.
I tried five random test cases. To scale this testing, one can add a way to directly read the testing csv file.
From there, one can compare the ouputs of the entire file with the predictions made by model.py. Since that is not
in scope, we will resort to doing five random test cases to demonstrate consistent functionality between the app and model.
'''

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import flask

#Load and split the data
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")
costa_rica_data = training_data.drop(['Target'], axis=1)
costa_rica_target = training_data['Target']
costa_rica_test = testing_data

#Clean the data to either replace or remove string columns
costa_rica_data.select_dtypes(exclude=[np.number]).head()
costa_rica_data = costa_rica_data.select_dtypes(include=[np.number], exclude=[np.object]).fillna(0)

costa_rica_test.select_dtypes(exclude=[np.number]).head()
costa_rica_test = costa_rica_test.select_dtypes(include=[np.number], exclude=[np.object]).fillna(0)

#Extract feature names
cols = costa_rica_data.columns.values
dicts = dict(zip(cols, range(len(cols))))

features = []
for key in dicts:
    #if dicts[key] in [98, 135, 134, 131, 118, 133, 109, 132, 94, 2]:
    if dicts[key] in [98, 135, 134, 131, 118, 133, 109, 132, 94, 2]:
        features.append(key)

#Define train and test data sets
X_train, X_test, y_train, y_test = train_test_split(costa_rica_data[features].values, costa_rica_target.values, test_size= 0.2, random_state=42)
X_predict = costa_rica_test[features].values

#Set up model for testing
clf = RandomForestClassifier(n_estimators=350, criterion='entropy', max_depth=15, random_state=42)
model = clf.fit(X_train, y_train)

#Use model to predict on testing_data set and perform integration testing_data
test_pred = model.predict(X_predict)

def integration_test():
#for sample 0, application predicted class 4, not vulnerable
if test_pred[0] == 4:
    return True

#for sample 251, application predicted class 4, not vulnerable
if test_pred[251] == 4:
    return True

#for sample 3369, application predicted class 4, not vulnerable
if test_pred[3369] == 4:
    return True

#for sample 23851, application predicted class 2, highly vulnerable
if test_pred[23851] == 2:
    return True

#for sample 23853, application predicted class 3, somewhat vulnerable
if test_pred[23853] == 3:
    return True

#for sample 23853, application predicted class 3, somewhat vulnerable
if test_pred[23853] == 3:
    return True

#for sample 51, application predicted class 1, extremely vulnerable
if test_pred[51] == 1:
    return True

#for sample 23757, application predicted class 1, extremely vulnerable
if test_pred[23757] == 1:
    return True

def print_results():
    if integration_test() == False:
        print('Test failed')
    else:
        print('Test passed')

print_results()
