# Predicting Poverty with Non-Econometric Factors
This project looks at a dataset based on Costa Rican neighborhoods to determine which common non-econometric factors can help best predict poverty. This can help donors allocate aid more efficiently and can help identify signs of poverty for early intervention. The final deliverable of this project is a Heroku hosted web application that allows interested donors to input relevant non-ecnometric features for a household into a form and returns the level of vulnerability to poverty that household is predicted to have. 

To view the application, go here: https://poverty-predictor.herokuapp.com/

The stages of this project included exploratory data analysis to get a sense of relevant features and relationships between features, building a classification machine learning model, exploring deep learning techniques and build scaled versions of both in order to handle large volumes of data that may arise in the future. The final stage involved developing the API and the actual interface for the web application.

## Exploratory Data Analysis
The analysis of the data showed that the size of a household, years of schooling and monthly rent payments correlated with certain levels of poverty. These factors, alongside with other factors made available about the conditions of the homes in these Costa Rican neighborhoods, can help us predict poverty using Machine Learning tactics. From the exploratory data analysis, the best features to predict poverty are related to education, number of children and size of the household.

## Machine Learning Prototype
A prototype was built using a pipeline to see which classification algorithms generated the highest accuracy (scoring was based on the accuracy_score function). The classification algorithms used were: DecisionTreeClassifier, RandomForestClassifier, XGBClassifier, LogisticRegression, KNeighborsClassifier and SVC.

The RandomForestClassifier had the highest accuracy score across all folds (5 folds were created using KFolds). Hyperparameter tuning was performed on certain parameters of the RandomForestClassifier model to determine the optimal configuration of the algorithm for this particular dataset. The RandomForestClassifier was fitted with a constrained max_depth of 15 to prevent overfitting. The number of trees, or n_estimators, was set to a very high number, 350, in order to get the best accuracy possible while minimizing overfitting.

## Deep Learning Prototype
An Artificial Neural Network was built using the Keras module. Prototyping was done using the keras wrapper that allows us to leverage scikit-learn. Twenty iterations of the neural network were performed in order to find the optimal batch_size, epochs, and activation functions, loss and optimizers. The use of GridSearchCV was avoided for the sake of learning purposes (to understand how different batch_size and epochs impacted the accuracy, in particular), but can be used if there are GPUs available to speed up the iterative prototyping of this neural network.

Ten folds were used (leveraging the same KFolds package that was used in the machine learning model) for cross-validation of the model. The model was quite well fitted, as the testing and training accuracy scores had less than a 2% delta. Further optimization can explored with the use of GPUs in order to speed up the learning process.

## Scaling with Dask
Dask was used to scale the machine learning prototype to make it faster on larger datasets (particularly the test dataset, which is 2.5 times larger than the train data set). The Dask module was used to recreate the same RandomForestClassifier algorithm that was successful in the final machine learning model, as well as the Artificial Neural Network that was successful in the final deep learning model.

## Application Deployment
This application was created with Flask and deployed through Heroku. The target audience for the initial phase of the application is foreign aid donors. The purpose of the application is to help donors determine how dire the need of a household is by using non-econometric factors to predict the household's vulnerability to poverty. There are certain restrictions around the types of values that one can input (must be positive float values) and some calculations happen in the backend to prevent the user from entering the same information in multiple ways (i.e. squaring a value, which can be done through calculations in the backend).

## Future Application Development
The future of this application includes adding tailored advice for the four different classes of need (extremely vulnerable, highly vulnerable, somewhat vulnerable, not vulnerable). This will make the information even more useful to donors because they can know what to do based on the prediction that comes from the data they put in. On top of this addition, the app can also suggest other households in a given area (which can be added to the form) that may be in more need of financial aid, if the household that the donor put in is Not Vulnerable or only Somewhat Vulnerable. The system can then automatically prioritize high needs households.

The next phase of this application is to integrate active learning. This would require us to add a database into the backend that can store new user inputted data, as well as the predictions that the machine learning model has made. Every three months, or certain number of submissions, we can validate the results of the model and see if the model needs to be retrained. If not, we can actively start inputting user submitted data so that it adds on to our data source, and retrain the model to improve the accuracy of the results, as well as help reduce overfitting. These two phases can build a more robust, user friendly and helpful aid allocation tool for donors.
