# Predicting Poverty with Non-Econometric Factors
This is my current project for Springboard. I am currently looking at a dataset based on Costa Rican neighborhoods to determine which common non-econometric factors can help best predict poverty. This can help donors allocate aid more efficiently and can help identify signs of poverty for early intervention.

## Data Analysis
The analysis of the data showed that the size of a household, years of schooling and monthly rent payments correlated with certain levels of poverty. These factors, alongside with other factors made available about the conditions of the homes in these Costa Rican neighborhoods, can help us predict poverty using Machine Learning tactics.

## Machine Learning Prototype
A prototype was built using a pipeline to see which classification algorithms generated the highest accuracy (scoring was based on the accuracy_score function). The classification algorithms used were: DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LogisticRegression, KNeighborsClassifier and SVC.

The RandomForestClassifier had the highest accuracy score across all folds (5 folds were created using KFolds). Hyperparameter tuning was performed on certain parameters of the RandomForestClassifier model to determine the optimal configuration of the algorithm for this particular dataset.

## Deep Learning Prototype
An Artificial Neural Network was built using the Keras module. Prototyping was done using the keras wrapper that allows us to leverage scikit-learn. Twenty iterations of the neural network were performed in order to find the optimal batch_size, epochs, and activation functions, loss and optimizers. The use of GridSearchCV was avoided for the sake of learning purposes (to understand how different batch_size and epochs impacted the accuracy, in particular), but can be used if there are GPUs available to speed up the iterative prototyping of this neural network.

Ten folds were used (leveraging the same KFolds package that was used in the machine learning model) for cross-validation of the model. The model was quite well fitted, as the testing and training accuracy scores had less than a 2% delta. Further optimization can explored with the use of GPUs in order to speed up the learning process.

## Scaling with Dask
Dask was used to scale the machine learning prototype to make it faster on larger datasets (particularly the test dataset, which is 2.5 times larger than the train data set). The Dask module was used to recreate the same RandomForestClassifier algorithm that was successful in the final machine learning model.

## Application Deployment
This application was deployed using Flask. The target audience for the initial phase of the application is foreign aid donors. The purpose of the application is to help donors determine how dire the need of a household is by using non-econometric factors to predict the household's vulnerability to poverty. The future of this application includes adding tailored advice for the four different classes of need (extremely vulnerable, highly vulnerable, somewhat vulnerable, not vulnerable). 
