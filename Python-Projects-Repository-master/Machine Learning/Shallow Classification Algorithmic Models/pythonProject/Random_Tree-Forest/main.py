# Random forests is a supervised learning algorithm. It can be used both for classification and regression.
# It is also the most flexible and easy to use algorithm. A forest is comprised of trees. It is said that the more
# trees it has, the more robust a forest is. Random forests creates decision trees on randomly selected data samples,
# gets prediction from each tree and selects the best solution by means of voting. It also provides a pretty good
# indicator of the feature importance.
# https://www.kaggle.com/tcvieira/simple-random-forest-iris-dataset

# Importing Libraries
import matplotlib
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inline
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Importing the Dataset
# Assign colum names to the dataset
colnames = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# Ρead data from CSV file into pandas dataframe
irisdata = pd.read_csv((
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\SVM Algorithm\iris.csv"),
    names=colnames)

# See the rows and columns of the data
irisdata.shape

# Depict how our dataset actually looks
irisdata.head()

# Divide the data into attributes and labels
X = irisdata.drop('species', axis=1)
y = irisdata['species']

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
accuracyScore = metrics.accuracy_score(y_test, y_pred)
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

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = accuracyScore
df_report['AUC'] = auc_score