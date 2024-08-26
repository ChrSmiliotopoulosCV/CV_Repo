# Our task is to predict whether a bank currency note is authentic or not based upon four attributes of the note i.e.
# skewness of the wavelet transformed image, variance of the image, entropy of the image, and curtosis of the image.
# This is a binary classification problem and we will use SVM algorithm to solve this problem. The rest of the
# section consists of standard machine learning steps.
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inline
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Ρead data from CSV file into pandas dataframe
bankdata = pd.read_csv("D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\SVM Algorithm/bill_authentication.csv")

# See the rows and columns of the data
bankdata.shape

# Depict how our dataset actually looks
bankdata.head()

# Divide the data into attributes and labels
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

# Divide data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Since we are going to perform a classification task, we will use the support vector classifier class,
# which is written as SVC in the Scikit-Learn's svm library. This class takes one parameter, which is the kernel type.
svclassifier = SVC(kernel='linear')
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

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

conf_matrix = confusion_matrix(y_test, y_pred)

# Evaluate the performance of SVM classifier with the use of classification_report() function.
print(classification_report(y_test, y_pred))

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(y_test, y_pred, output_dict=True)

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = svm_score
# df_report['cv_scores'] = cv_scores
# df_report['mean_top_performing'] = mean_top_performing

# Extract fn, fp, tn, tp from confusion matrix and add as columns to the dataframe.
df_report['FN'] = fn
df_report['FP'] = fp
df_report['TN'] = tn
df_report['TP'] = tp

# Measure and compare our classifiers’ performance by calculating the area under the curve (AUC) which will
# result in a score called AUROC (Area Under the Receiver Operating Characteristics). A perfect AUROC should
# have a score of 1 whereas a random classifier will have a score of 0.5.
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)
plot_roc_curve(svclassifier, X_test, y_test)
print(plot_roc_curve(svclassifier, X_test, y_test))
plt.show()
auc = np.trapz(fpr, tpr)
print(auc)
df_report['AUC'] = auc