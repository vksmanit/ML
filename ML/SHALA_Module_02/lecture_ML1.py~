#!/usr/bin/python3 

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the machine learning packages
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# =====================================================================================================
# Overfitting Model : BEGIN {
# =====================================================================================================

# Loading the candy data from an url
candy_url = "https://raw.githubusercontent.com/shala2020/shala2020.github.io/master/Lecture_Materials/Google_Colab_Notebooks/MachineLearning/L1/candy-data.csv"
candy_data = pd.read_csv(candy_url)

# Printing the first few rows of the candy data 
#print(candy_data.head())

# Extracting the features and label
X = candy_data.drop(['winpercent', 'competitorname'], axis = 1)
y = candy_data['winpercent']

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)

# Instantiating the classifier

#rfr = RandomForestRegressor() # this will also work
rfr = RandomForestRegressor(n_estimators=25,
                            random_state=1111,
                            max_features=4)

# Fitting the instantiator
rfr.fit(X_train, y_train)

# Printing the training and testing accuracies 
print('The training error is {0:.2f}'.format(mean_absolute_error(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(mean_absolute_error(y_test, rfr.predict(X_test))))
print(rfr.get_params())


# =====================================================================================================
# Since Training and Testing Errors are 3.99 and 8.57 --> hence we can say that this is overfitting model
# Overfitting Model END }
# =====================================================================================================

# =====================================================================================================
# Visualization of Parameter change  
# =====================================================================================================

