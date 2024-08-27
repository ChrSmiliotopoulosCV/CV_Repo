# K-means Clustering:
# The goal of the K-means clustering algorithm is to find groups in the data, with the number of groups represented
# by the variable K. The algorithm works iteratively to assign each data point to one of the K groups based on the
# features that are provided.
# The outputs of executing a K-means on a dataset are:
# K centroids: Centroids for each of the K clusters identified from the dataset.
# Labels for the training data: Complete dataset labelled to ensure each data point is assigned to one of the clusters.

# Decision Tree Classifier
# The general motive of using a Decision Tree is to create a training model which can be used to predict the class or
# value of target variables by learning decision rules inferred from prior data(training data).

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings('ignore')
from pandas.plotting import andrews_curves
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Importing the Dataset
# Assign column names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Ρead data from CSV file into pandas dataframe
irisdata = pd.read_csv((
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\SVM Algorithm\iris.csv"),
    names=colnames)

# See the rows and columns of the data
irisdata.shape

# Depict how our dataset actually looks
irisdata_head = irisdata.head()
irisdata_descr = irisdata.describe()
print(irisdata_head)
print(irisdata_descr)

# Visualizing the data using matplotlib
irisdata.plot(kind="scatter", x="sepal-length", y="sepal-width")
plt.show()

# Visualizing the data using pandas’ andrew curves
andrews_curves(irisdata.drop("sepal-length", axis=1), "Class")
plt.show()

# Divide the data into attributes and labels
# Last column values excluded
x = irisdata.iloc[:, :-1].values
# Last column value
y = irisdata.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Decision Tree Classifier
# The general motive of using a Decision Tree is to create a training model which can be used to predict the class or
# value of target variables by learning decision rules inferred from prior data(training data).
classifier = DecisionTreeClassifier()
# Training the classifier
classifier.fit(x_train, y_train)
# Making predictions
y_pred = classifier.predict(x_test)
# Summary of the predictions made by the   classifier
print(classification_report(y_test, y_pred))
# Evaluate the quality of the output
print(confusion_matrix(y_test, y_pred))
# Accuracy score
print('Accuracy is', accuracy_score(y_pred, y_test))
svm_score = accuracy_score(y_pred, y_test)

# Evaluate the performance of SVM classifier with the use of confusion matrix.
print("Confusion_matrix:")
print(confusion_matrix(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)


# Function to obtain TP, FN FP, and TN for each class in the confusion matrix of our multiclass problem
def counts_from_confusion(conf_matrix):
    """
    Obtain TP, FN FP, and TN for each class in the confusion matrix
    """

    counts_list = []

    # Iterate through classes and store the counts
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]

        fn_mask = np.zeros(conf_matrix.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(conf_matrix, fn_mask))

        fp_mask = np.zeros(conf_matrix.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(conf_matrix, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(conf_matrix, tn_mask))

        counts_list.append({'Class': i,
                            'TP': tp,
                            'FN': fn,
                            'FP': fp,
                            'TN': tn})
    return counts_list


# TP, FN FP, and TN variable storing and print()
countlist = counts_from_confusion(conf_matrix)
print(countlist)

# TP, FN FP, and TN DataFrame() creation and Index[] manipulation
df_confusion_countlist = pd.DataFrame(countlist)
df_confusion_countlist.Class[df_confusion_countlist.Class == 0] = 'Iris-setosa'
df_confusion_countlist.Class[df_confusion_countlist.Class == 1] = 'Iris-versicolor'
df_confusion_countlist.Class[df_confusion_countlist.Class == 2] = 'Iris-virginica'

# Evaluate the performance of SVM classifier with the use of classification_report() function.
print(classification_report(y_test, y_pred))


# LabelBinarizer() function in order to evaluate the AUC ROC score for our multi-class problem
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


# auc_score calculation and print()
auc_score = multiclass_roc_auc_score(y_test, y_pred)
print(auc_score)

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(y_test, y_pred, output_dict=True)

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(y_test, y_pred, output_dict=True)

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = svm_score
df_report['AUC'] = auc_score

