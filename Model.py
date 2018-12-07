# -*- coding: utf-8 -*-
"""
Minimizing churn rate through analysis of financial habits

@author: Ivana Hybenova
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import random
import time



### Data Preprocessing ###

dataset = pd.read_csv('new_churn_data.csv')

# Saving user id
user_identifier = dataset['user']
dataset = dataset.drop(columns = ['user'])

# One Hot Encoding
dataset = pd.get_dummies(dataset)
dataset.columns
# manual dropping one of the categories of each categorical variable
dataset = dataset.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])


# Splitting into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'),
                                                    dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)

# Balancing the training set
y_train.value_counts()

# Storing column names
X_train_column_names = X_train.columns.values 

# Undersampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
X_train = pd.DataFrame(X_resampled)
y_train = pd.Series(y_resampled)

# Restoring column names
X_train.columns = X_train_column_names

'''
 M O D E L   T R A I N I N G
'''
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization

# Scikit-Learn for Modeling
import sklearn

# Pickle for saving model files
import pickle

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import support vector classifier
from sklearn.svm import SVC 

# Import RandomForestClassifier and GradientBoostingClassifer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Function for splitting training and test set
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+

# Function for balancing the classes
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# For data preprocessing
from sklearn import preprocessing

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Classification metrics (added later)
from sklearn.metrics import roc_curve, auc

# Import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score





# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
features = X_train.columns.values
importance = model.feature_importances_
# create a data frame with feature importance
features_importance = pd.DataFrame({'feature':features, 'importance':importance})
# order by importance
features_importance = features_importance.sort_values(by=['importance'], ascending=False)

# Forward selection 1 round
X_train_1round = X_train[["purchases_partners", "age", "reward_rate", "cc_recommended"]]
X_test_1round = X_test[["purchases_partners", "age", "reward_rate", "cc_recommended"]]
# Random forest as testing classifier for feature selection
model = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(random_state=123))
model.fit(X_train_1round, y_train)
pred1 = model.predict(X_test_1round)

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred1)
accuracy_score(y_test, pred1)      # 0.66

# Forward selection 2 round
X_train_2round = X_train[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin"]]
X_test_2round = X_test[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin"]]
# Random forest as testing classifier for feature selection
model.fit(X_train_2round, y_train)
pred2 = model.predict(X_test_2round)

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred2)
accuracy_score(y_test, pred2)    # 0.676

# Forward selection 3 round
X_train_3round = X_train[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits"]]
X_test_3round = X_test[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits"]]
# Random forest as testing classifier for feature selection
model.fit(X_train_3round, y_train)
pred3 = model.predict(X_test_3round)
probabilities3 = model.predict_proba(X_test_3round)


# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred3)
accuracy_score(y_test, pred3)    # 0.6824

# Forward selection 4 round
X_train_4round = X_train[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits", "purchases"]]
X_test_4round = X_test[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits", "purchases"]]
# Random forest as testing classifier for feature selection
model.fit(X_train_4round, y_train)
pred4 = model.predict(X_test_4round)

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred4)
accuracy_score(y_test, pred4)    # 0.671

# Forward selection 5 round
X_train_5round = X_train[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "purchases"]]
X_test_5round = X_test[["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "purchases"]]
# Random forest as testing classifier for feature selection
model.fit(X_train_5round, y_train)
pred5 = model.predict(X_test_5round)

# Display confusion matrix for y_test and pred
confusion_matrix(y_test, pred5)
accuracy_score(y_test, pred5)    # 0.677

# Creating new X_train with only picked predictors columns
predictors = ["purchases_partners", "age", "reward_rate", "cc_recommended", "cc_application_begin", "deposits"]
X_train = X_train[predictors]

# Build model pipelines =======================================================
# Pipeline dictionary
pipelines = {
    'lr' : make_pipeline(preprocessing.StandardScaler(), LogisticRegression(random_state=123)),
    'svc' : make_pipeline(preprocessing.StandardScaler(), SVC(random_state=123, probability=True)),      
    'rf' : make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(random_state=123)),
    'gb' : make_pipeline(preprocessing.Normalizer(), GradientBoostingClassifier(random_state=123))
}

# List tuneable hyperparameters of our Logistic pipeline
pipelines['lr'].get_params()

# Logistic Regression hyperparameters   #  Higher C means weaker penalty
lr_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-3, 1e3, 10),
    'logisticregression__penalty' : ('l1', 'l2')
}

# List tuneable hyperparameters of SVC
pipelines['svc'].get_params()

# Support vector classifier hyperparameters
svc_hyperparameters = {
    'svc__C' : [0.1, 1, 10, 100],  # higher means higher penalty, prone to overfitting
    'svc__gamma': [1, 0.1, 0.01, 0.001], # large gamma means closer data points have higher weights, prone to overfitting
    'svc__kernel': ['rbf', 'linear']
}

# List tuneable hyperparameters of RF
pipelines['rf'].get_params()

# Random Forest hyperparameters
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 200, 500, 1000],
    'randomforestclassifier__max_features': ['auto',  0.33]
}

# List tuneable hyperparameters of GB
pipelines['gb'].get_params()

# Boosted Tree hyperparameters
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
    'gradientboostingclassifier__max_depth': [1, 3, 5]
}

# Create hyperparameters dictionary
hyperparameters = {
    'lr' : lr_hyperparameters,
    'svc' : svc_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

# Fit & tune models with cross-validation
# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1, verbose = 8)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')

'''
W I N N E R   S E L E C T I O N
'''
# Evaluate metrics ===========================================================
# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )                #rf: 0.698, gb: 0.687
    
# Display best parameters for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_params_ )
    
# building Confusion matrix
for name, model in fitted_models.items():
    # Coumpute the predictions
    pred = fitted_models[name].predict(X_test)
    
    # Display confusion matrix for y_test and pred
    cm = confusion_matrix(y_test, pred)

    sns.heatmap(cm, annot = True)
    print(classification_report(y_test,pred))
    
    # Area under ROC curve
    # Area under ROC curve is the most reliable metric for classification tasks.
    # Area under ROC curve is equivalent to the probability that a randomly chosen '0' 
    # observation ranks higher (has a higher predicted probability) than a randomly chosen '1' observation.
    # Basically, it's saying... if you grabbed two observations and exactly one of them was 
    # the positive class and one of them was the negative class, what's the likelihood that 
    # your model can distinguish the two?
        
    # building ROC curve
    # Calculate ROC curve from y_test and pred
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]

    fpr, tpr, thresholds = roc_curve(y_test, pred)
    
    # Store fpr, tpr, thresholds in DataFrame and display last 10
    pd.DataFrame({'FPR': fpr, 'TPR' : tpr, 'Thresholds' : thresholds}).tail(10)
     
    # Initialize figure
    fig = plt.figure(figsize=(8,8))
    plt.title('Receiver Operating Characteristic')
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label='l1')
    plt.legend(loc='lower right')
    
    # Diagonal 45 degree line
    plt.plot([0,1],[0,1],'k--')
    
    # Axes limits and labels
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    # AUROC - area under the ROC curve
    # Remember, that AUROC is equivalent to the probability that a randomly chosen '0' observation 
    # ranks higher (has a higher predicted probability) than a randomly chosen '1' observation.
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    
    # Calculate AUROC
    print( auc(fpr, tpr) )
    
'''
Model training round 2
'''
# Build model pipelines =======================================================
# Pipeline dictionary
pipelines = { 
    'rf' : make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(random_state=123)),
    'gb' : make_pipeline(preprocessing.Normalizer(), GradientBoostingClassifier(random_state=123))
}


# List tuneable hyperparameters of RF
pipelines['rf'].get_params()

# Random Forest hyperparameters
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [1000, 1500, 2000],
    'randomforestclassifier__max_features': ['auto',  0.33]
}

# List tuneable hyperparameters of GB
pipelines['gb'].get_params()

# Boosted Tree hyperparameters
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [200, 400, 600],
    'gradientboostingclassifier__learning_rate': [0.08, 0.1, 0.3],
    'gradientboostingclassifier__max_depth': [4,5,6]
}

# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters
}

# Fit & tune models with cross-validation
# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1, verbose = 8)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')

'''
W I N N E R   S E L E C T I O N
'''
# Evaluate metrics ===========================================================
# Display best_score_ for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_score_ )
    
# Display best parameters for each fitted model
for name, model in fitted_models.items():
    print( name, model.best_params_ )

# Picking the winner
for name, model in fitted_models.items():
    pred = model.predict_proba(X_test)
    pred = [p[1] for p in pred]
    
    fpr, tpr, thresholds = roc_curve(y_test, pred)
    print( name, auc(fpr, tpr) )
    
# Save winning model as final_model.pkl
with open('final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)
    
# If we output that object directly, we can also see the winning values for our hyperparameters.
fitted_models['rf'].best_estimator_


'''
5. P R O J E C T   D E L I V E R Y
'''
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
# pd.options.mode.chained_assignment = None  # default='warn'

# Pickle for reading model files
import pickle

# Scikit-Learn for Modeling
import sklearn
from sklearn.model_selection import train_test_split # Scikit-Learn 0.18+

# Area under ROC curve - if we don't need to plot the ROC curve
from sklearn.metrics import roc_auc_score

# Load final_model.pkl as model
with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

# 1. Confirm your model was saved correctly ===================================
# Display model object
model











