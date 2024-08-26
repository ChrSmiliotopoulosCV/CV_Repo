##############################################################################################
######################## IsolationForest Categorical Unsupervised ML #########################
##############################################################################################
# To start, import the following libraries:

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc)
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import sklearn
import math 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

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
path = r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\Prints\One SVM\OneSVMLog01.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        # Multiclass Scenario
        # r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
        # Binary Scenario
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\binary-csv-csv[BinaryFeaturesReduced].csv"
    ),
    encoding="ISO-8859-1",
    low_memory=False,
)

# Check the validity of the dataset
df.head()
# print("The dataset's first 5 rows are: ")
# print(df.head())
# print("The dataset's last 5 rows are: ")
# print(df.tail())

# Count the values of each Label in the Dataset
print(
    "Each Label within the full-csv(Evaluated Titles).csv is comprised from the following elements: " 
)
# print(df["Label"].value_counts())
print(df["Label"].value_counts())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
# print(df)
# X = df.drop("Label", axis=1)
# y = df["Label"]
X = df.drop("Label", axis=1)
y = df["Label"]

# Check the validity of the X, y subsets
# print("The subset's (X,y) first 5 rows are: ")
# print(X.head())
# print(y.head())

# print("The subset's (X,y) last 5 rows are: ")
# print(X.tail())
# print(y.tail())

# print(df.info())   # all non-null and numeric [except the labels]
# print(X.info())
# print(y.info())

# Divide dataframe into training and test sets.
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.5, test_size=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

print("The subset's (y_test) counted values are: ")
print(y_test.value_counts())

# Next, we create an instance of the IsolationForest class.
# model = IsolationForest(n_estimators=100, verbose=99, random_state=42)
# model = LocalOutlierFactor(n_neighbors=3, contamination='auto', novelty=True)
# model = LocalOutlierFactor()
model = OneClassSVM(gamma='auto', verbose=99)

# We train the model.
model.fit(X_train, y_train)

# We predict the data in the test set.
y_pred = model.predict(X_test)
print(y_pred)

# The IsolationForest assigns a value of -1 instead of 0. Therefore, we replace it to ensure we only have 2 distinct values in our confusion matrix.
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
# y_pred = y_pred.replace(['old value'], 'new value')
print(y)

# As we can see, the algorithm does a good job of predicting what data points are anomalous.
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# print(classification_report(y_test, y_pred, labels=[0, 1, 2], digits=4))
print(classification_report(y_test, y_pred, digits=4))
# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:", accuracy_score(y_test, y_pred))

# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)
y_pred = lb.transform(y_pred)
    
print("ROC AUC score:", roc_auc_score(y_test, y_pred, average="macro"))

cm = conf_matrix

sns.set(font_scale=1.75)
# Labels for Binary Classification
x_axis_labels = ["0", "1"]  # labels for x-axis
y_axis_labels = ["0", "1"]  # labels for y-axis
p = sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 22}
)
p.xaxis.tick_top()  # x-axis on top
p.xaxis.set_label_position("top")
p.tick_params(length=0)
p.set(
    xlabel="Predicted label",
    ylabel="True label",
    xticklabels=x_axis_labels,
    yticklabels=y_axis_labels,
)

plt.show()

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the Multiclass experiment was: ")
print("--- %s seconds ---" % (timeDuration))

