import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#Load and split the data
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")
costa_rica_data = training_data.drop(['Target'], axis=1)
costa_rica_target = training_data['Target']

#Clean the data to either replace or remove string columns
costa_rica_data.select_dtypes(exclude=[np.number]).head()
costa_rica_data = costa_rica_data.select_dtypes(include=[np.number], exclude=[np.object]).fillna(0)

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

#Fit our Random Forest model to the data
clf = RandomForestClassifier(n_estimators=350, criterion='entropy', max_depth=15, random_state=42)
model = clf.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
val_score = model.score(X_test, y_test)
print('Random Forest Train Accuracy: '+str(round(train_score*100,2))+'%')
print('Random Forest Validation Accuracy: '+str(round(val_score*100,2))+'%')
print('Train/Test Delta: '+str(round((train_score - val_score)*100,2))+'%')

#Save the trained model as a pickle string, load pickled model + use it to make predictions
pickled_model = open('model.pkl', 'wb')
saved_model = pickle.dump(model, pickled_model)
pickled_model.close()
