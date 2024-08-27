import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import (
#     accuracy_score,
#     plot_roc_curve,
#     roc_curve,
#     plot_confusion_matrix,
#     auc,
#     roc_auc_score
# )
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc)
from sklearn.metrics import precision_recall_fscore_support as score
from pyod.models.knn import KNN

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
path = r"/Users/christossmiliotopoulos/Desktop/testIFLogfile.txt"
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
        # r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/binary-csv-csv[BinaryFeaturesReduced].csv"
        r"/Users/christossmiliotopoulos/Desktop/lmd2023EvalSubset_01.csv"
    ),
    encoding="ISO-8859-1",
    low_memory=False,
)

# Check the validity of the dataset
df.head()
print("The dataset's first 5 rows are: ")
print(df.head())
print("The dataset's last 5 rows are: ")
print(df.tail())

# Count the values of each Label in the Dataset
print(
    "Each Label within the full-csv(Evaluated Titles).csv is comprised from the following elements: "
)
print(df["Label"].value_counts())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
# print(df)
X = df.drop("Label", axis=1)
y = df["Label"]

# Check the validity of the X, y subsets
print("The subset's (X,y) first 5 rows are: ")
print(X.head())
print(y.head())

print("The subset's (X,y) last 5 rows are: ")
print(X.tail())
print(y.tail())

print(df.info())   # all non-null and numeric [except the labels]
print(X.info())
print(y.info())

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

# The two dataframes, entitled X_test and y_test dataframes respectively, are concatenated with the pd.concat() function of the pandas framework. 
result = pd.concat([X_test, y_test], axis=1)
print(result)
print(len(result))
print(result.columns)
print(result["Label"].value_counts())

# # Implemeting from the sklearn.neighbors the NearestNeighbors Unsupervised Algorithm. 
# knn = KNN(contamination=0.03,method='mean',n_neighbors=5) 
knn = KNN(contamination=0.3, method='mean',n_neighbors=12) 
knn.fit(X_train)

predicted = pd.Series(knn.predict(X_test),index=X_test.index) 
print('Number of outliers = ', predicted.sum())

print("#######################################################################################################################")
print("## The results of Multiclass/Binary Scenarios from the evaluation of the X, y fullsets or their equivalent subsets. ###")
print("#######################################################################################################################")

# uclf = nbrs.fit(X_train, y_train)
# pred = uclf.predict(X_test)
# print(pred)

# print() of the True and Predicted values confusion_matrix().
conf_matrix = confusion_matrix(y_test, predicted)
print(confusion_matrix(y_test, predicted))
# print(classification_report(y_test, kmeans_predict, labels=[0, 1, 2], digits=4))
print(classification_report(y_test, predicted, labels=[0, 1], digits=4))
# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:", accuracy_score(y_test, predicted))

# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y_test)
y_test1 = lb.transform(y_test)
y_pred = lb.transform(predicted)
    
print("ROC AUC score:", roc_auc_score(y_test1, y_pred, average="macro"))

cm = conf_matrix

# sns.set(font_scale=1.75)
# # Labels for Multiclass Classification
# x_axis_labels = ["0", "1", "2"]  # labels for x-axis
# y_axis_labels = ["0", "1", "2"]  # labels for y-axis
# p = sns.heatmap(
#     cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 22}
# )
# p.xaxis.tick_top()  # x-axis on top
# p.xaxis.set_label_position("top")
# p.tick_params(length=0)
# p.set(
#     xlabel="Predicted label",
#     ylabel="True label",
#     xticklabels=x_axis_labels,
#     yticklabels=y_axis_labels,
# )

# plt.show()
    
# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the Multiclass experiment was: ")
print("--- %s seconds ---" % (timeDuration))