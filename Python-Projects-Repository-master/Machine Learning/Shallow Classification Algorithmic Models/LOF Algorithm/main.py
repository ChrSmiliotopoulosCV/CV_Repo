# https://github.com/pratikmishra356/ML-Credit_Card_fraud_detection/blob/master/credit_card_fraud_detection.ipynb

# Importing Libraries
import data as data
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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Importing the Dataset

# Ρead data from CSV file into pandas dataframe
data = pd.read_csv((
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\LOF Algorithm\creditcard.csv"))

# Depict how our dataset actually looks
data = data.sample(frac=0.1, random_state=1)
print(data.shape)
data.head(3)

# Store data columns as a list of attributes
columns = data.columns.tolist()

# Distinguish data in dataset into X = data[columns] and Y = data[target]
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
X = data[columns]
Y = data[target]

# Class attribute's data labelling into Fraud and valid - outlier_factor calculation
Fraud = data[data["Class"] == 1]
valid = data[data["Class"] == 0]
outlier_fraction = len(Fraud) / float(len(valid))

# Create lof classifier variable
lof = LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)

# Fit data and calculate accuracy_score - classification_report - confusion_matrix
y_pred = lof.fit_predict(X)
scores_pred = lof.negative_outlier_factor_
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
n_errors = (y_pred != Y).sum()
print("{}: {}".format('Local Outliar Factor', n_errors))
print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))
print(confusion_matrix(Y, y_pred))
tn, fp, fn, tp = confusion_matrix(Y, y_pred).ravel()

conf_matrix = confusion_matrix(Y, y_pred)

# Accuracy Score - Implementation of Scikit-Learn's function, accuracy_score, which accepts the true value and
# predicted value as its input to calculate the accuracy score of a model.

lof_score = accuracy_score(Y, y_pred)
print("Accuracy score (LOF): ", lof_score)

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(Y, y_pred, output_dict=True)

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = lof_score

# Extract fn, fp, tn, tp from confusion matrix and add as columns to the dataframe.
df_report['FN'] = fn
df_report['FP'] = fp
df_report['TN'] = tn
df_report['TP'] = tp

# Measure and compare our classifiers’ performance by calculating the area under the curve (AUC) which will
# result in a score called AUROC (Area Under the Receiver Operating Characteristics). A perfect AUROC should
# have a score of 1 whereas a random classifier will have a score of 0.5.
fpr, tpr, thresholds = metrics.roc_curve(Y, y_pred, pos_label=0)
auc = np.trapz(fpr, tpr)
print(auc)
df_report['AUC'] = auc