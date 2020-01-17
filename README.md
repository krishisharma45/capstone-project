# Predicting Poverty with Non-Econometric Factors
This is my current project for Springboard. I am currently looking at a dataset based on Costa Rican neighborhoods to determine which common non-econometric factors can help best predict poverty. This can help donors allocate aid more efficiently and can help identify signs of poverty for early intervention.

## Data Analysis
The analysis of the data showed that the size of a household, years of schooling and monthly rent payments correlated with certain levels of poverty. These factors, alongside with other factors made available about the conditions of the homes in these Costa Rican neighborhoods, can help us predict poverty using Machine Learning tactics.

## Machine Learning Prototype
A prototype was built using a pipeline to see which classification algorithms generated the highest accuracy (scoring was based on the accuracy_score function). The classification algorithms used were: DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LogisticRegression, KNeighborsClassifier and SVC.

The DecisionTreeClassifier had the highest accuracy score across all folds (5 folds were created using KFolds). Hyperparameter tuning was performed on certain parameters of the DecisionTreeClassifier model to determine the optimal configuration of the algorithm for this particular dataset.

## Scaling with Pyspark
Pyspark was used to scale the machine learning prototype to make it faster on larger datasets (particularly the test dataset, which is 2.5 times larger than the train data set). The MLlib library of Spark was used to recreate the same DecisionTreeClassifier algorithm that was successul in the prototyping stages.

----------------------------------------------------------------------

The future of this project involves performing Deep Learning techniques to see if we can get better predictions and to create an API for deployment to a web application.
