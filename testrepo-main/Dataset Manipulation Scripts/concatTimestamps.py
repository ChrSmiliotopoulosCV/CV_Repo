import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import (
    accuracy_score,
    plot_roc_curve,
    roc_curve,
    plot_confusion_matrix,
    auc,
    roc_auc_score
)

import time
import sys

import warnings
warnings.filterwarnings("ignore")

# The class Logger() is responsible for recording the terminal's screen in the pre-defined with the
# path variable destination folder. The format of the file will be .txt.

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


start_time = time.time()

# The path of the destination folder on which the logs.txt file will be created.
path = r"C:\Users\chrsm\Desktop\Timestamps\timestampsLogs01.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df0 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Timestamps\chris0.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

df1 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Timestamps\chris1.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

df2 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Timestamps\chris2.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

df3 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Timestamps\chris3.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

df4 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Timestamps\chris4.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

# The two dataframes, entitled X_test and y_test dataframes respectively, are concatenated with the pd.concat() function of the pandas framework. 
result1 = pd.concat([df0, df1, df2], axis=0)
print(result1)

print(len(result1))
print(result1.columns)
# print(result["date1"].value_counts())

result1.to_csv('concatedChrisNew1.csv')

result2 = pd.concat([df3, df4], axis=0)
print(result2)

print(len(result2))
print(result2.columns)
# print(result["date1"].value_counts())

result2.to_csv('concatedChrisNew2.csv')

df5 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\concatedChrisNew1.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

df6 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\concatedChrisNew2.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

resultFinal = pd.concat([df5, df6], axis=0)
print(resultFinal)

print(len(resultFinal))
print(resultFinal.columns)
# print(result["date1"].value_counts())

resultFinal.to_csv('concatedChrisFinal.csv')