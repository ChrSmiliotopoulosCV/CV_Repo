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
path = r"C:\Users\chrsm\Desktop\AddColumn\addColumnLogs01.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df1 = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\AddColumn\concatedChrisFinal.csv"
    ),
    encoding="utf-8",
    low_memory=False,
)

# df2 = pd.read_csv(
#     (
#         r"C:\Users\chrsm\Desktop\AddColumn\dfLMD23.csv"
#     ),
#     encoding="utf-8",
#     low_memory=False,
# )

# print(df1.head())
print(df1.head())

# df1.to_csv('df1.csv')
df1.to_csv('concatedChrisFinal(addedColumn)).csv')