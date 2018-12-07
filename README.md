# Minimizing-churn-rate-through-analysis-of-financial-habits
Predict which users are likely to churn, so that the company can focus on re-engaging these users with the product.

Data: 
Mock-up dataset based on trends found in real world case studies; 27 000 instances and 30 features (40 after creating dummy variables of categorical ones)

Goal: 
Predict which users are likely to churn, so that the company can focus on re-engaging these users with the product.
Challenges: Imbalanced classes (solved by under sampling), high dimensionality (solved by computing feature importance with Extra trees classifier and backwards elimination)

Algorithms: 
LR regularization L1 and L2, SVM classifier, GB classifier, RF classifier

Measures: 
Confusion metrics, Area under the ROC curve

Project delivery: 
Python script executing locally hosted flask api, that takes in raw data, preprocess them (feature engineering, standardization), do the predictions and provide downloadable zipped .xlsx file with 3 columns: user identifier, probability of class 1 and predicted class

Files: 
EDA.py - Python script that contains exploration of the data 
Model.py - Python script that contains data preprocessing, training and tuning the algorithms and saving the pipeline with final model. flask_predict_api.py - Python scirpt with the application

churn_data.csv - Dataset provided for the project 
new_churn_data.csv - Dataset after the EDA
raw_unseen_data.csv - This is actually hold out set (test.set) after the split I saved for being able to test the app 


Instructions: Download  raw_unseen_data.csv, all zip files (after unzipping make sure to have final_model.pkl in separate "model" folder created among the other downloaded files) and flask_predict_api.py.

Through your command line navigate to the folder you are storing these files. Make sure you have python path in your enviroment variables and run command python flask_predict_api.py

From your browser navigate to your localhost on port 8000. Click on predict_api and then try it out!. Insert raw_unseen_data and press execute. After some time scroll down and click on Download the zip.file, which contains the predictions.
