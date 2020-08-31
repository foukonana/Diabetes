# Dimensionality reduction as an alternative to feature selection 

import os
import pandas as pd 
from sklearn.decomposition import PCA


# load the data
path = os.getcwd() + '\datasets'
dbDF = pd.read_csv(path + '\pima_indians_diabetes.csv')

# split the data in predictors and the class to be predicted
X = dbDF.values[:, 0:8]
Y = dbDF.values[:, 8]

pca = PCA(n_components=3)
fit = pca.fit(X)

print(f'Explained variance of each pca component {fit.explained_variance_ratio_} with a total of {round(fit.explained_variance_ratio_.sum()*100, 2)}%.')
print(f'The pca components are {fit.components_}')
