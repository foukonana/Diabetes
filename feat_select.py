# Feature selection on pima indians diabetes datas

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import os

# load the data
path = os.getcwd() + '\datasets'
dbDF = pd.read_csv(path + '\pima_indians_diabetes.csv')

# the last column of the dataframe 'Outcome' is the class we want to predict
# 1 when the subject is diabetic, 0 when it is not

# split the data in predictors and the class to be predicted
X = dbDF.values[:, 0:8]
Y = dbDF.values[:, 8]


# part 1 
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
test = SelectKBest(score_func=f_classif, k=4)
fit = test.fit(X, Y)
features = fit.transform(X)
print(features[0:5, :])


# part 2 - Recursive feature elimination 
# https://machinelearningmastery.com/rfe-feature-selection-in-python/
model = LogisticRegression(solver = 'lbfgs')
rfe = RFE(estimator = model, n_features_to_select=3)
fit = rfe.fit(X,Y)
print(f'RFE selected {rfe.n_features_} features.')
print(f'Selected features using RFE estimator are {[x for x,y in zip(dbDF.columns.tolist()[:8], fit.support_.tolist()) if y]}')


# part 3 - Bagged decision trees
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X,Y)
importances = model.feature_importances_.tolist()
print(f'The 3 most important features using bagged decision trees are {[x for x,y in zip(dbDF.columns.tolist()[:8], importances) if y in sorted(importances, reverse=True)[:3]]}')


# part 4 - Lasso regularization 
# lasso regression has embeded feature selection 
# usually, it is a good alternative when a linear model is overfitting
params = {'alpha':[1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}
lasso = Lasso()
lasso_regressor = GridSearchCV(lasso, params, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X,Y)

print(f'Lasso regularization best parameter {lasso_regressor.best_params_}')
print(f'Lasso regularization best score {lasso_regressor.best_score_}')
