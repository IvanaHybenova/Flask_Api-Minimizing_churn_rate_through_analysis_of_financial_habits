# -*- coding: utf-8 -*-
"""
Minimizing churn rate through analysis of financial habits - EDA

Ivana Hybenova
"""

#### Importing Libraries ####

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('churn_data.csv')

# Putting aside raw_data for testing created app
# Splitting into Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'),
                                                    dataset['churn'],
                                                    test_size = 0.2,
                                                    random_state = 0)

X_test.to_csv('raw_data.csv', index = False)

### EDA ###

dataset.head()
dataset.columns
dataset.describe()


## Cleaning Data

# Removing NaN
dataset.isna().sum()
# removing 4 rows with missing age
dataset = dataset[pd.notnull(dataset['age'])] 
# removing columns with too many NAs
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned']) 
dataset.isna().any()

'''
## Histograms
# creating temporary dataset with only numerical columns
dataset2 = dataset.drop(columns = ['user', 'churn'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 5, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False) # getting rid of y labels
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())

    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
'''
## Pie charts - to make sure that the other value for binary columns have both possible outcomes for resposnse variable (they are useful)
dataset2.dtypes
dataset2 = dataset[['housing', 'is_referred', 'app_downloaded',
                    'web_user', 'app_web_user', 'ios_user',
                    'android_user', 'registered_phones', 'payment_type',
                    'waiting_4_loan', 'cancelled_loan', 'received_loan',
                    'rejected_loan', 'zodiac_sign', 'left_for_two_month_plus',
                    'left_for_one_month']]

dataset2 = dataset.drop(columns = ['user', 'churn'])

fig = plt.figure(figsize=(15, 20))
plt.suptitle('Pie Chart Distributions', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(6, 3, i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False) # getting rid of y labels
    f.set_title(dataset2.columns.values[i - 1])
    values = dataset2.iloc[:, i-1].value_counts(normalize = True).values # normalize to get percentages
    index = dataset2.iloc[:, i-1].value_counts(normalize = True).index 
    
    plt.pie(values, labels = index, autopct = '%1.1f%%')
    plt.axis('equal') # nor x or y are shown in the image
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

## Exploring uneven features
# checking whether uneven distributions have both values of target variable
dataset[dataset2.waiting_4_loan ==1].churn.value_counts()
dataset[dataset2.cancelled_loan ==1].churn.value_counts()
dataset[dataset2.received_loan ==1].churn.value_counts()
dataset[dataset2.rejected_loan ==1].churn.value_counts()
dataset[dataset2.left_for_one_month ==1].churn.value_counts()
dataset[dataset2.registered_phones ==5].churn.value_counts()

## Correlation with Response Variable (Note: Models like RF are not linear like these)
dataset.drop(columns = ['churn', 'user', 'housing',    # dropping categorical variables
                        'payment_type', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(
    figsize = (20,10), title = 'Correlation with the response variable', fontsize = 15,
    rot = 45, grid = True)

## Correlation Matrix
sn.set(style="white")

# Compute the correlation matrix
corr = dataset.drop(columns = ['user', 'churn']).corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# dropping the columns that causes multicolinearity
dataset = dataset.drop(columns = ['app_web_user', 'android_user'])

dataset.to_csv('new_churn_data.csv', index = False)


