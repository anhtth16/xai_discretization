'''
Bias - Variance Decomposition
- Dataset: adult
- Discretization: Supervised: Decision Tree
- Model: KNN-Hamming
- Updated: 18/03/2023

Process:
- Load pre-trained model (skops)
- Run bias-variance decomposition
- Save result to "adult_evaluation_dt_knn.txt"
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
os.chdir(f'{cwd}/adult_sup')
print("New working directory:", os.getcwd())
new_wd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(os.getcwd()))

#------------ MAIN SCRIPT -----------------#
#-------- 2. Decision Tree Discretization -----------
#-------- 2.1 DT, max_depth = 2

# Read data
df_dt1 = pd.read_csv('DT_small_discretized_adult.csv')
df_dt1.rename(columns={'class':'label'}, inplace=True)

disc = 'DT'
max_depth = 2

df_dt1.info()
data = df_dt1.values
data.shape

features = df_dt1.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify = Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_dt1[features].nunique()

# SMOTE-Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'adult'
discretizer = 'DT'
disc_param = 'max_depth = 2'

model_name = f"{dataset}_{model}_{discretizer}_{max_depth}.skops"

print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/adult_evaluation_dt_knn.txt"
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
print(f'Execution time {model}- default, {disc}, max_depth = {max_depth} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

#-------- 2.1 DT, max_depth = 3
# Complete code for data preperation
# Read data
df_dt2 = pd.read_csv('DT_medium_discretized_adult.csv')
df_dt2.rename(columns={'class':'label'}, inplace=True)

disc = 'DT'
max_depth = 3

df_dt2.info()
data = df_dt2.values
data.shape

features = df_dt2.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify = Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_dt2[features].nunique()

# SMOTE-Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'adult'
discretizer = 'DT'
disc_param = 'max_depth = 3'

model_name = f"{dataset}_{model}_{discretizer}_{max_depth}.skops"

print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/adult_evaluation_dt_knn.txt"
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
print(f'Execution time {model}- default, {disc}, max_depth = {max_depth} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

#-------- 2.1 DT, max_depth = 4
# Complete code for data preperation
# Read data
df_dt3 = pd.read_csv('DT_large_discretized_adult.csv')
df_dt3.rename(columns={'class':'label'}, inplace=True)

disc = 'DT'
max_depth = 4

df_dt3.info()
data = df_dt3.values
data.shape

features = df_dt3.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify = Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_dt3[features].nunique()

# SMOTE-Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'adult'
discretizer = 'DT'
disc_param = 'max_depth = 4'

model_name = f"{dataset}_{model}_{discretizer}_{max_depth}.skops"

print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/adult_evaluation_dt_knn.txt"
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
print(f'Execution time {model}- default, {disc}, max_depth = {max_depth} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

#-------- 2.1 DT, max_depth = 5
# Complete code for data preperation
# Read data
df_dt4 = pd.read_csv('DT_verylarge_discretized_adult.csv')
df_dt4.rename(columns={'class':'label'}, inplace=True)

disc = 'DT'
max_depth = 5

df_dt4.info()
data = df_dt4.values
data.shape

features = df_dt4.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify = Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_dt4[features].nunique()

# SMOTE-Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'adult'
discretizer = 'DT'
disc_param = 'max_depth = 5'

model_name = f"{dataset}_{model}_{discretizer}_{max_depth}.skops"

print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/adult_evaluation_dt_knn.txt"
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
print(f'Execution time {model}- default, {disc}, max_depth = {max_depth} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')
print('---DONE SRUN!!!---')