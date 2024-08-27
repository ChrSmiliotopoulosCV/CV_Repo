# Importing Libraries
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve

import sys

# The class Logger() is responsible for recording the terminal's screen in the pre-defined with the
# path variable destination folder. The format of the file will be .txt.

class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()

# The path of the destination folder on which the logs.txt file will be created.
path = (r"C:\Users\chrsm\Desktop\ExtractedFiles\Swallow_Classification\DecisionTree\decisionTreeLogs.txt")
sys.stdout = Logger(path)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv((r"C:\Users\chrsm\Desktop\ExtractedFiles\Finally_Processed_Dataset\ohePreProcessedDataset.csv"), encoding="ISO-8859-1", low_memory=False)
df.head()
print(df.head())

# Investigate the Label column with value_counts. The statement that follows will return the total 
# values representing each category in the 'Label' column.
print("The sum of the sumerical representation of each feature of the 'Label' column has as follows:")
print(df['Label'].value_counts())
print(df.columns)
print(len(df.columns))

# isnull() check for the df dataframe
print(df.isnull().sum())

# Function to check the prevalence of the 'Label' column of the df dataframe
def calc_prevalence(y):
 return (sum(y)/len(y))

print(calc_prevalence(df["Label"].values))

# We are now ready to split our samples and train a model!
# ####################################################################################
# Split Samples
# ####################################################################################
# For simplicity, I’ll just split into two datasets: train (70%) and validation (30%). 
# It is important to shuffle your samples because you may have been given the data in 
# order of dates.
# ####################################################################################

# shuffle the samples
df = df.sample(n = len(df), random_state = 42)
# df = df.reset_index(drop = True)
df_valid = df.sample(frac = 0.3, random_state = 42)
print(df_valid['Label'].value_counts())
df_train = df.drop(df_valid.index)
print(df_train['Label'].value_counts())

# We can check the prevalence is about 20% in each:
print('Valid prevalence(n = %d):%.3f'%(len(df_valid), calc_prevalence(df_valid['Label'].values)))
print('Train prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train['Label'].values)))

# We can now build our X (inputs) and Y(output) for training and validation:
X_train = df_train.values
X_valid = df_valid.values
y_train = df_train['Label'].values
y_valid = df_valid['Label'].values
print('Training shapes:',X_train.shape, y_train.shape)
print('Validation shapes:',X_valid.shape, y_valid.shape)

# Train a Machine Learning Model
# #################################### Pay Attention for Future Work!!! ####################################
# Since the focus of this post is about the datetime features, we will just train a random forest model here. 
# Note if you want to use other types of models, you may need to scale or normalize your data. Another thing 
# you may want to do is convert the dayofweek into a categorical variable via one-hot encoding. We don’t need 
# to do these things for a tree-based method though.

rf = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
rf.fit(X_train, y_train)

# We can then get our predictions with:
y_train_preds = rf.predict_proba(X_train)[:,1]
print(y_train_preds)
y_valid_preds = rf.predict_proba(X_valid)[:,1]
print(y_valid_preds)

# Here we will evaluate performance of the model. For that reason we created the two functions 
# calc_specificity() and print_report().
def calc_specificity(y_actual, y_pred, thresh):
    # calculates specificity
    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh), average=None)
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f'%auc)
    print('accuracy:%.3f'%accuracy)
    print('recall:%.3f'%recall)
    print('precision:%.3f'%precision)
    print('specificity:%.3f'%specificity)
    print('prevalence:%.3f'%calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity

# We set the threshold to be the same as the prevalence.
thresh = 1.887

print(' ')
print('Random Forest')
print('Training:')
# print_report(y_train, y_train_preds, thresh)
print(accuracy_score(y_train, y_train_preds))
print('Validation:')
# print_report(y_valid, y_valid_preds, thresh)