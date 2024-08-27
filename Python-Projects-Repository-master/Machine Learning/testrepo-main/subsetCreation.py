import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler


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


pd.options.display.max_rows = 999
start_time = time.time()

# The path of the destination folder on which the logs.txt file will be created.
path = r"/Users/christossmiliotopoulos/Desktop/Machine Learning Metrics for Network Datasets Evaluation/NDVM/subsetReport.txt"
# path = r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\DNN_Experiments\pexMLPlogfile(UML)03.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)
print(pd.__version__)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/binary-csv-csv[BinaryFeaturesReduced].csv"
        # r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\binary-csv-csv[BinaryFeaturesReduced].csv"
        # r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
        # r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
    ),
    encoding="ISO-8859-1",
    low_memory=False,
)
df.head()
print(df)
print(
    "Each Label within the binary-csv-csv[BinaryFeaturesReduced].csv is comprised from the "
    "following elements:"
)
print(df["Label"].value_counts())
print(df.isnull().sum())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
# print(df.info())
print(df.head())
X = df.drop("Label", axis=1)
y = df["Label"]

# Divide dataframe into training and test sets.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.025, test_size=0.012)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

print(X_test.head())
print(y_test.head())

X_test['Label'] = y_test
print(X_test.head())

print(type(X_test))

df1 = X_test
print(df1.head())

X_test.to_csv('subset.csv')