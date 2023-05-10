'''
Bias - Variance Decomposition
- Dataset: pageblock
- Discretization: unsupervised: EWD
- Model: KNN-Hamming
- Updated: 22/03/2023

Process:
- Load pre-trained model (skops)
- Run bias-variance decomposition
- Save result to "pageblock_evaluation_ewd_knn.txt"
'''
# Import 
import pandas as pd
import numpy as np

import skops.io as sio
import mlxtend
from collections import Counter

# For model evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import zero_one_loss
#This library is used to decompose bias and variance in our models
from mlxtend.evaluate import bias_variance_decomp
import warnings
warnings.filterwarnings('ignore')

import six
import sys
sys.modules['sklearn.externals.six'] = six

# Import 
import skops.io as sio
import joblib
import mlxtend

#------------ CHECK WORKING DIRECTORY -----------------#
import os
cwd = os.getcwd()
print("Working directory:", cwd)

# Change the current working directory
os.chdir(f'{cwd}/pageblock')
print("New working directory:", os.getcwd())
new_wd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

#------------ MAIN SCRIPT -----------------#
#------------ 1. EWD -----------------#
# 1.1 EWD, k = 4

# Read data
df_ewd1 = pd.read_csv('pageblock_ewd1.csv')
df_ewd1.drop(df_ewd1.columns[0], axis=1, inplace = True)
df_ewd1.rename(columns={'class':'label'}, inplace=True)

disc = 'EWD'
k = 4

df_ewd1.info()
data = df_ewd1.values
data.shape

features = df_ewd1.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]
print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify=Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_ewd1[features].nunique()

# SMOTE-Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
k=4
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'EWD'
disc_param = 'k = 4'
model_name = f"{dataset}_{model}_{discretizer}_{k}.skops"
print(model_name)
path = f"{new_wd}/{model_name}"
print("Path to load model: ", path)
loaded_knn = sio.load('pageblock_knn_EWD_4.skops', trusted = True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ewd_knn.txt"
f = open(f_path, "a")
import time
start = time.time() # For measuring time execution

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
loaded_knn, x_resample, y_resample, x_test, y_test,
loss='0-1_loss',
random_seed=123)

# Write result
print(f'Evaluation result: {model}, {discretizer}, {disc_param}', file = f)
print('Average expected loss: %.3f' % avg_expected_loss, file = f)
print('Average bias: %.3f' % avg_bias, file = f)
print('Average variance: %.3f' % avg_var, file = f)
print('Sklearn 0-1 loss: %.3f' % zero_one_loss(y_test,y_pred_knn), file = f)

end = time.time()
print(f'Execution time {model}- default, {disc}, k = {k} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

##-------- 1.2 EWD, k = 7
# Read data
df_ewd2 = pd.read_csv('pageblock_ewd2.csv')
df_ewd2.drop(df_ewd2.columns[0], axis=1, inplace = True)
df_ewd2.rename(columns={'class':'label'}, inplace=True)

disc = 'EWD'
k = 7
df_ewd2.info()
data = df_ewd2.values
data.shape

features = df_ewd2.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]
print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify=Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_ewd2[features].nunique()

# SMOTE Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
k=7
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'EWD'
disc_param = 'k = 7'
model_name = f"{dataset}_{model}_{discretizer}_{k}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ewd_knn.txt"
f = open(f_path, "a")
import time
start = time.time() # For measuring time execution

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
loaded_knn, x_resample, y_resample, x_test, y_test,
loss='0-1_loss',
random_seed=123)
#---
print(f'Evaluation result: {model}, {discretizer}, {disc_param}', file = f)
print('Average expected loss: %.3f' % avg_expected_loss, file = f)
print('Average bias: %.3f' % avg_bias, file = f)
print('Average variance: %.3f' % avg_var, file = f)
print('Sklearn 0-1 loss: %.3f' % zero_one_loss(y_test,y_pred_knn), file = f)

end = time.time()
print(f'Execution time {model}- default, {disc}, k = {k} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

##--------  1.3 EWD, k = 10

# Read data
df_ewd3 = pd.read_csv('pageblock_ewd3.csv')
df_ewd3.drop(df_ewd3.columns[0], axis=1, inplace = True)
df_ewd3.rename(columns={'class':'label'}, inplace=True)

disc = 'EWD'
k = 10

df_ewd3.info()
data = df_ewd3.values
data.shape

features = df_ewd3.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify=Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_ewd3[features].nunique()

# SMOTE - Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
k=10
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'EWD'
disc_param = 'k = 10'
model_name = f"{dataset}_{model}_{discretizer}_{k}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ewd_knn.txt"
f = open(f_path, "a")
import time
start = time.time() # For measuring time execution

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
loaded_knn, x_resample, y_resample, x_test, y_test,
loss='0-1_loss',
random_seed=123)
#---
print(f'Evaluation result: {model}, {discretizer}, {disc_param}', file = f)
print('Average expected loss: %.3f' % avg_expected_loss, file = f)
print('Average bias: %.3f' % avg_bias, file = f)
print('Average variance: %.3f' % avg_var, file = f)
print('Sklearn 0-1 loss: %.3f' % zero_one_loss(y_test,y_pred_knn), file = f)

end = time.time()
print(f'Execution time {model}- default, {disc}, k = {k} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

print('---DONE SRUN!!!---')