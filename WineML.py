import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = pd.read_csv(r"C:\Users\advay\Downloads\winequality-white.csv", sep=';')

print("\nDescription of data in csv:")
print (data.describe())

# train test split
y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# preprocess and scale data
scaler = preprocessing.StandardScaler().fit(X_train) #create scaler
X_train_scaled = scaler.transform(X_train)
print("\nMean of scaled training data:")
print (X_train_scaled.mean(axis=0))
print("\nStandard Dev of scaled training data:")
print (X_train_scaled.std(axis=0))

# transform test set using training set scale
X_test_scaled = scaler.transform(X_test)
print("\nMean of scaled test data:")
print (X_test_scaled.mean(axis=0))
print("\nStandard Dev of scaled test data:")
print (X_test_scaled.std(axis=0))

# create pipeline
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# setting up hyperparameters
print("\nParameters:")
print (pipeline.get_params())
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# cross validation pipeline with SciKit
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)  # fit and tune model            

print("\nBest Parameters:")
print(clf.best_params_) # best params

print("\nRefit pipline with better hyperparameters:")
print(clf.refit) # refit on best hyperparameters

# using pipeline on test data
y_pred = clf.predict(X_test)

print("\nR Squared Score:")
print(r2_score(y_test, y_pred))

print("\nMean Squared Error:")
print(mean_squared_error(y_test, y_pred))
# results outputted
