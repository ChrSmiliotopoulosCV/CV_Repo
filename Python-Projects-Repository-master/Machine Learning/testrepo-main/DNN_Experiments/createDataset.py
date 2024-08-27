# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################
# ################ Pay Attention!!! This script has been created for benchmarking reasons only. ################
# ##############################################################################################################
# ##############################################################################################################
# ##############################################################################################################

import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler

# Create an imbalanced dataset
X, y = make_classification(n_samples=100000, n_features=32, n_informative=32,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.995, 0.005],
                           class_sep=0.5, random_state=0)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_normal = X_train[np.where(y_train == 0)]

print(np.count_nonzero(X_train))
print(np.count_nonzero(X_train_normal))
print(np.count_nonzero(X_test))

# data1 = {
#   "age": [50, 40, 30, 40, 20, 10, 30],
#   "qualified": [True, False, False, False, False, True, True]
# }

# data2 = {
#   "label": [0, 1, 0, 0, 0, 0, 1]
# }
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# newdf = df1.where(df2["label"] == 1)
# print(newdf)
# newdf = newdf.dropna()
# print(newdf)

