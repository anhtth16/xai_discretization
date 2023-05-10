'''
Bias - Variance Decomposition
- Dataset: pageblock
- Discretization: unsupervised: FFD
- Model: KNN-Hamming
- Updated: 22/03/2023

Process:
- Load pre-trained model (skops)
- Run bias-variance decomposition
- Save result to "pageblock_evaluation_ffd_knn.txt"
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
#------------ 3. FFD -----------------#
##--------  3.1 FFD, m = 10

# Complete code for data preperation
# Read data
df_ffd1 = pd.read_csv('pageblock_ffd1.csv')
df_ffd1.drop(df_ffd1.columns[0], axis=1, inplace = True)
df_ffd1.rename(columns={'class':'label'}, inplace=True)

disc = 'ffd'
m = 10

df_ffd1.info()
data = df_ffd1.values
data.shape

features = df_ffd1.drop('label', axis = 1).columns

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
n_categories = df_ffd1[features].nunique()

# SMOTE - Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'FFD'
disc_param = 'm = 10'

model_name = f"{dataset}_{model}_{discretizer}_{m}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ffd_knn.txt"
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
print(f'Execution time {model}- default, {disc}, m = {m} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

#--------  3.2 FFD, m = 30

# Complete code for data preperation
# Read data
df_ffd2 = pd.read_csv('pageblock_ffd2.csv')
df_ffd2.drop(df_ffd2.columns[0], axis=1, inplace = True)
df_ffd2.rename(columns={'class':'label'}, inplace=True)

disc = 'ffd'
m = 30

df_ffd2.info()
data = df_ffd2.values
data.shape

features = df_ffd2.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]
#X = df_ffd2[features]
#Y = df_ffd2['class']

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify=Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_ffd2[features].nunique()

# SMOTE - Tomek
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'FFD'
disc_param = 'm = 30'

model_name = f"{dataset}_{model}_{discretizer}_{m}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
#loaded_knn = sio.load('pageblock_knn_FFD_30.skops', trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ffd_knn.txt"
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
print(f'Execution time {model}- default, {disc}, m = {m} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

##--------  3.3 FFD, m = 60

# Read data
df_ffd3 = pd.read_csv('pageblock_ffd3.csv')
df_ffd3.drop(df_ffd3.columns[0], axis=1, inplace = True)
df_ffd3.rename(columns={'class':'label'}, inplace=True)

disc = 'ffd'
m = 60

df_ffd3.info()
data = df_ffd3.values
data.shape

features = df_ffd3.drop('label', axis = 1).columns

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
n_categories = df_ffd3[features].nunique()

# SMOTE - Tomek:
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'FFD'
disc_param = 'm = 60'

model_name = f"{dataset}_{model}_{discretizer}_{m}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
#loaded_knn = sio.load('pageblock_knn_FFD_60.skops', trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ffd_knn.txt"
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
print(f'Execution time {model}- default, {disc}, m = {m} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')

##--------  3.4 FFD, m = 100

# Complete code for data preperation
# Read data
df_ffd4 = pd.read_csv('pageblock_ffd4.csv')
df_ffd4.drop(df_ffd4.columns[0], axis=1, inplace = True)
df_ffd4.rename(columns={'class':'label'}, inplace=True)

disc = 'ffd'
m = 100

df_ffd4.info()
data = df_ffd4.values
data.shape

features = df_ffd4.drop('label', axis = 1).columns

# separate the data into X and y
X = data[:, : len(features)]
Y = data[:,-1]
#X = df_ffd4[features]
#Y = df_ffd4['class']

print(X.shape, Y.shape)

# Split train test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state = 30, stratify=Y)

# Check representation of class
print('Class representation - original: ', Counter(Y)) 
print('Class representation - training data: ', Counter(y_train)) 
print('Class representation - testing data: ', Counter(y_test)) 

# Check number of categories for features
n_categories = df_ffd4[features].nunique()

# SMOTE - Tomek:
from imblearn.combine import SMOTETomek
smt_tomek = SMOTETomek(random_state=42)
x_resample, y_resample = smt_tomek.fit_resample(x_train, y_train)

# Load models
model = 'KNN-Hamming'
dataset = 'pageblock'
discretizer = 'FFD'
disc_param = 'm = 100'

model_name = f"{dataset}_{model}_{discretizer}_{m}.skops"
print(model_name)
loaded_knn = sio.load(model_name, trusted=True)
#loaded_knn = sio.load('pageblock_knn_FFD_100.skops', trusted=True)
y_pred_knn = loaded_knn.predict(x_test)

# Decomposition
f_path = f"{new_wd}/pageblock_evaluation_ffd_knn.txt"
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
print(f'Execution time {model}- default, {disc}, m = {m} is: {end - start}.', file = f) # Total time execution
print('=='*20, file = f)
f.close()

print(f'Done srun {model_name}')
print('---DONE SRUN!!!---')