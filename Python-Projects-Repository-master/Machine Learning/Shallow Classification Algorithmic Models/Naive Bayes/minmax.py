# Naive Bayes is the most straightforward and fast classification algorithm, which is suitable for a large chunk of
# data. Naive Bayes classifier is successfully used in various applications such as spam filtering,
# text classification, sentiment analysis, and recommender systems. It uses Bayes theorem of probability for
# prediction of unknown class.

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
from sklearn.naive_bayes import GaussianNB

# Importing the Dataset
# # Assign colum names to the dataset
# colnames = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# Ρead data from CSV file into pandas dataframe
oheAlgorithm_Dataframe = pd.read_csv((
    r"C:\Users\chrsm\Desktop\AWID3 Project Tasks [14-11-21]\minmaxAlgo_Dataframe_[final].csv"))
# oheAlgorithm_Dataframe = oheAlgorithm_Dataframe.drop(oheAlgorithm_Dataframe.columns[[0, 1, 2]], axis=1)

# See the rows and columns of the data
oheAlgorithm_Dataframe.shape

# Depict how our dataset actually looks
oheAlgorithm_Dataframe.head()

# Divide the data into attributes and labels
X = oheAlgorithm_Dataframe.drop('Label', axis=1)
y = oheAlgorithm_Dataframe['Label']

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Gaussian NB classifier
nb = GaussianNB()

# Train the model using the training sets y_pred=nb.predict(X_test)
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

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
df_confusion_countlist.Class[df_confusion_countlist.Class == 0] = 'Flooding'
df_confusion_countlist.Class[df_confusion_countlist.Class == 1] = 'Impersonation'
df_confusion_countlist.Class[df_confusion_countlist.Class == 2] = 'Normal'
# # df_confusion_countlist.Class[df_confusion_countlist.Class == 3] = 'Evil_Twin'
# # df_confusion_countlist.Class[df_confusion_countlist.Class == 4] = 'Kr00k'
# # df_confusion_countlist.Class[df_confusion_countlist.Class == 5] = 'Krack'
# # df_confusion_countlist.Class[df_confusion_countlist.Class == 6] = 'Normal'
# # df_confusion_countlist.Class[df_confusion_countlist.Class == 7] = 'Rogue_AP'


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
