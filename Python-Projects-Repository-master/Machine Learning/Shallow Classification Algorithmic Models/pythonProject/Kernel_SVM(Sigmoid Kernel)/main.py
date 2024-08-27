# Our task is to predict whether a bank currency note is authentic or not based upon four attributes of the note i.e.
# skewness of the wavelet transformed image, variance of the image, entropy of the image, and curtosis of the image.
# This is a binary classification problem and we will use SVM algorithm to solve this problem. The rest of the
# section consists of standard machine learning steps
# (https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/),
# .
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

# Importing the Dataset
# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Ρead data from CSV file into pandas dataframe
irisdata = pd.read_csv((
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\SVM Algorithm\iris.csv"),
    names=colnames)

# See the rows and columns of the data
irisdata.shape

# Depict how our dataset actually looks
irisdata.head()

# Divide the data into attributes and labels
X = irisdata.drop('Class', axis=1)
y = irisdata['Class']

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Since we are going to perform a classification task, we will use the support vector classifier class,
# which is written as SVC in the Scikit-Learn's svm library. This class takes one parameter, which is the kernel type.
# In the case of polynomial kernel, you also have to pass a value for the degree parameter of the SVC class.
# This basically is the degree of the polynomial.
svclassifier = SVC(kernel='sigmoid', degree=8)
svclassifier.fit(X_train, y_train)

# Making predictions
y_pred = svclassifier.predict(X_test)

# Accuracy Score - Implementation of Scikit-Learn's function, accuracy_score, which accepts the true value and
# predicted value as its input to calculate the accuracy score of a model.

svm_score = accuracy_score(y_test, y_pred)
print("Accuracy score (SVM): ", svm_score)

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
df_report['model accuracy'] = svm_score
df_report['AUC'] = auc_score
