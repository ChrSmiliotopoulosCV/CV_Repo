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

start_time = time.time()

df = pd.read_csv(r"C:\Users\chrsm\Desktop\Iris.csv") # reading the data 

df.head() # first 5 rows

print(df.head()) # print the datasets first 5 rows

df.drop(["Id"],axis=1,inplace=True)    # dropped

df.tail()   # last 5 rows

print(df.tail())

print(df.info())   # all non-null and numeric [except the labels]

# sns.pairplot(data=df,hue="Species",palette="Set2")
# plt.show()

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
# print(df)
X = df.drop("Species", axis=1)
y = df["Species"]

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

# #################### Important Comment!!! ####################
# We see that iris-setosa is easily separable from the other two. Especially when we can see in different colors for corresponding Labels like above.
# But our mission was finding the Labels that we didn't knew at all, So Let's create a suitable scenario.

# # Adjusting the Dataset for Unsupervised Learning
# # I will simply do not use labels column on my "new" Dataset

# features = df.loc[:,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]

# print(features)

# From now on, we don't know the real labels or amount of labels anymore (Shhh!)

# Implemeting the K Means Clustering
kmeans = KMeans(n_clusters=99)    #### 99 CLUSTERS ????? WHY 99 ????? Because I don't know the right amount of Labels. Don't worry, There is a solution for it. #####

# Finding the best amount of clusters to get most accurate results (KMeans)
# I will use ELBOW RULE, which is basically looking for a plot line that respectively has a slope nearest to 90 degrees compared to y axis and be smallest possible. 
# (yes, looks like an elbow)

wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
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

# 3 or 2 seems to be our Best value(s) for clusters. (By the Elbow Rule)

print(X)
kmeans
kmeans = KMeans(n_clusters=3)
kmeans_predict = kmeans.fit_predict(X)

print(kmeans_predict)

# cross tabulation table for kmeans
df1 = pd.DataFrame({'labels':kmeans_predict,"Species":df['Species']})
ct1 = pd.crosstab(df1['labels'],df1['Species'])

print(ct1.head())

# # from sklearn.cluster import AgglomerativeClustering

# # # hierarchy
# # hc_cluster = AgglomerativeClustering(n_clusters=3)
# # hc_predict = hc_cluster.fit_predict(features)

# # # cross tabulation table for Hierarchy
# # df2 = pd.DataFrame({'labels':hc_predict,"Species":df['Species']})
# # ct2 = pd.crosstab(df2['labels'],df2['Species'])


plt.figure(figsize=(24,8))
plt.suptitle("CROSS TABULATIONS",fontsize=18)
plt.subplot(1,2,1)
plt.title("KMeans")
sns.heatmap(ct1,annot=True,cbar=False,cmap="Blues")

# # plt.subplot(1,2,2)
# # plt.title("Hierarchy")
# # sns.heatmap(ct2,annot=True,cbar=False,cmap="Blues")

plt.show()

print("##################################################################################")
print("################## K-Means Multiclass Scenario over Subsets ######################")
print("##################################################################################")
# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

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

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the Multiclass experiment was: ")
print("--- %s seconds ---" % (timeDuration))