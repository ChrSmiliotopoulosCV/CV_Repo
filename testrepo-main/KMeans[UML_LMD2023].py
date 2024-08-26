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
path = r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\Prints\KNNfullLogsMulticlassFeaturesReduced01.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        # Multiclass Scenario
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
        # Binary Scenario
        # r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\binary-csv-csv[BinaryFeaturesReduced].csv"
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

# # Implemeting the K Means Clustering
kmeans = KMeans(n_clusters=99)    #### 99 CLUSTERS ????? WHY 99 ????? Because I don't know the right amount of Labels. Don't worry, There is a solution for it. #####

# ######################## Elbow Method for the Best ''k'' ########################
# Finding the best amount of clusters to get most accurate results (KMeans)
# I will use ELBOW RULE, which is basically looking for a plot line that respectively has a slope nearest to 90 degrees compared to y axis and be smallest possible. 
# (yes, looks like an elbow)

wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_test)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(20,8))
plt.title("WCSS / K Chart", fontsize=18)
plt.plot(range(1,15),wcss,"-o")
plt.grid(True)
plt.xlabel("Amount of Clusters",fontsize=14)
plt.ylabel("Inertia",fontsize=14)
plt.xticks(range(1,20))
plt.tight_layout()
plt.show()

# ######################## K-Means Initialization of Parameters ########################
# Define and Initialize the kmeans algorithmic model
kmeans = KMeans(n_clusters=3, verbose=1, algorithm="elkan", tol=0.001, max_iter=1000, )
# kmeans = KMeans(n_clusters=3, verbose=1)

print("##################################################################################")
print("#################### K-Means Binary Scenario over Subsets ########################")
print("##################################################################################")
for n in range(10):
    kmeansfit = kmeans.fit(X_train, y_train)
    kmeans_predict = kmeans.predict(X_test)
    # kmeans_predict = kmeans.fit_predict(X)
    print(kmeans_predict)
    # print() of the True and Predicted values confusion_matrix().
    conf_matrix = confusion_matrix(y_test, kmeans_predict)
    print(confusion_matrix(y_test, kmeans_predict))
    print(classification_report(y_test, kmeans_predict, labels=[0, 1, 2], digits=4))
    # print(classification_report(y_test, kmeans_predict, labels=[0, 1], digits=4))
    # The Accuracy of the sklearn metrics is printed on the terminal's screen.
    print("Accuracy:", accuracy_score(y_test, kmeans_predict))
    # The AUC score is calculated and printed.

    # function for scoring roc auc score for multi-class
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(kmeans_predict)
    
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

print("##################################################################################")
print("#### The results of Binary Scenario from the evaluation of the X, y fullsets #####")
print("##################################################################################")

for n in range(10):
    kmeansfit = kmeans.fit(X_train, y_train)
    kmeans_predict = kmeans.predict(X)
    print(kmeans_predict)
    # print() of the True and Predicted values confusion_matrix().
    conf_matrix = confusion_matrix(y, kmeans_predict)
    print(confusion_matrix(y, kmeans_predict))
    # print(classification_report(y_test, kmeans_predict, labels=[0, 1, 2], digits=4))
    print(classification_report(y, kmeans_predict, labels=[0, 1], digits=4))
    # The Accuracy of the sklearn metrics is printed on the terminal's screen.
    print("Accuracy:", accuracy_score(y, kmeans_predict))

    # function for scoring roc auc score for multi-class
    lb = LabelBinarizer()
    lb.fit(y)
    y_test1 = lb.transform(y)
    y_pred = lb.transform(kmeans_predict)
    
    print("ROC AUC score:", roc_auc_score(y_test1, y_pred, average="macro"))

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
print("The time duration of the Binary experiment was: ")
print("--- %s seconds ---" % (timeDuration))

print("##################################################################################")
print("################## K-Means Multiclass Scenario over Subsets ######################")
print("##################################################################################")
kmeansfit = kmeans.fit(X_train, y_train)
kmeans_predict = kmeans.predict(X_test)
# kmeans_predict = kmeans.fit_predict(X)
print(kmeans_predict)
# print() of the True and Predicted values confusion_matrix().
conf_matrix = confusion_matrix(y_test, kmeans_predict)
print(confusion_matrix(y_test, kmeans_predict))
print(classification_report(y_test, kmeans_predict, labels=[0, 1, 2], digits=4))
# print(classification_report(y_test, kmeans_predict, labels=[0, 1], digits=4))
# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:", accuracy_score(y_test, kmeans_predict))
# The AUC score is calculated and printed.

# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)
y_pred = lb.transform(kmeans_predict)
    
print("ROC AUC score:", roc_auc_score(y_test, y_pred, average="macro"))

cm = conf_matrix

sns.set(font_scale=1.75)
# Labels for Multiclass Classification
x_axis_labels = ["0", "1", "2"]  # labels for x-axis
y_axis_labels = ["0", "1", "2"]  # labels for y-axis
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


print("##################################################################################")
print("## The results of Multiclass Scenario from the evaluation of the X, y fullsets ###")
print("##################################################################################")

kmeansfit = kmeans.fit(X_train, y_train)
kmeans_predict = kmeans.predict(X)
print(kmeans_predict)
# print() of the True and Predicted values confusion_matrix().
conf_matrix = confusion_matrix(y, kmeans_predict)
print(confusion_matrix(y, kmeans_predict))
# print(classification_report(y_test, kmeans_predict, labels=[0, 1, 2], digits=4))
print(classification_report(y, kmeans_predict, labels=[0, 1], digits=4))
# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:", accuracy_score(y, kmeans_predict))

# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y)
y_test1 = lb.transform(y)
y_pred = lb.transform(kmeans_predict)
    
print("ROC AUC score:", roc_auc_score(y_test1, y_pred, average="macro"))

cm = conf_matrix

sns.set(font_scale=1.75)
# Labels for Multiclass Classification
x_axis_labels = ["0", "1", "2"]  # labels for x-axis
y_axis_labels = ["0", "1", "2"]  # labels for y-axis
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