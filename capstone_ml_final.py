import pandas as pd
import numpy as np
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Load and split the data
training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")
costa_rica_data = training_data.drop(['Target'], axis=1)
costa_rica_target = training_data['Target']

#Clean the data to either replace or remove string columns
costa_rica_data.select_dtypes(exclude=[np.number]).head()
costa_rica_data = costa_rica_data.select_dtypes(include=[np.number], exclude=[np.object]).fillna(0)

#Split data into 80% train, 20% validation split
X_train, X_test, y_train, y_test = train_test_split(costa_rica_data.values, costa_rica_target.values, test_size= 0.2, random_state=42)

#Fit a Decision Tree with hyperparameters to get a baseline idea of performance
clf = DecisionTreeClassifier(criterion='gini', max_depth=75, random_state=42)
model = clf.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print('Decision Tree Train Accuracy: '+str(round(train_score*100,2))+'%')
print('Decision Tree Test Accuracy: '+str(round(test_score*100,2))+'%')

#Run Feature Importance to extract relevant features
features = model.feature_importances_
features_dict = dict(zip(costa_rica_data.columns.values, features))
ranked_features_df = pd.DataFrame(features_dict,
                                  index=['Feature Importance']).T.sort_values('Feature Importance', ascending=False)[:10]

#Fit Decision Tree with these most important features
X_train, X_test, y_train, y_test = train_test_split(costa_rica_data[ranked_features_df.index].values, costa_rica_target.values, test_size= 0.2, random_state=42)
clf = DecisionTreeClassifier(criterion='gini', max_depth=60)
model = clf.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
val_score = model.score(X_test, y_test)
print('Decision Tree Train Accuracy: '+str(round(train_score*100,2))+'%')
print('Decision Tree Validation Accuracy: '+str(round(val_score*100,2))+'%')

#Save the model to disk
filename = 'capstone_ml_final.sav'
pickle.dump(model, open(filename, 'wb'), protocol=2)

#Load the model from the disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)
