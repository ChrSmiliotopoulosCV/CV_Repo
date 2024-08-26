# Importing Libraries

from catboost import CatBoostClassifier
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_curve, plot_confusion_matrix, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
import lightgbm as lgb
import seaborn as sns
import time
import sys

# The class Logger() is responsible for recording the terminal's screen in the pre-defined with the
# path variable destination folder. The format of the file will be .txt.

class Logger:
 
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()

start_time = time.time()

# The path of the destination folder on which the logs.txt file will be created.
path = (r"C:\Users\chrsm\Desktop\ExtractedFiles [18 Jan 2023]\Swallow_Classification_Results\BaggingClassifier\2nd Set of Experiments [ExtendedDF]\baggingClfLogs(cb).txt")
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv((r"C:\Users\chrsm\Desktop\ExtractedFiles [18 Jan 2023]\newExtendedSchema\Preprocessing_Dataframes [Final Extraction]\1.8M_FinallySelected\Best Results\full-csv.csv"), encoding="ISO-8859-1", low_memory=False)
df.head()
print(df)
print('Each Label within the oheMinMaxPreProcessedDataset.csv is comprised from the following elements: ')
print(df['Label'].value_counts())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac = 1)
print(df)
X = df.drop('Label', axis=1)
y = df['Label']

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print('The length of the X_train set is: ', len(X_train))
print('The length of the X_test set is: ', len(X_test))
print('The length of the y_train set is: ', len(y_train))
print('The length of the y_test set is: ', len(y_test))

# Bootstrap Bagging Aggregation algorithm. Methods such as Decision Trees, can be prone to overfitting on the training set which can lead to 
# wrong predictions on new data. Bootstrap Aggregation (bagging) is a ensembling method that attempts to resolve overfitting for classification 
# or regression problems. Bagging aims to improve the accuracy and performance of machine learning algorithms. It does this by taking random 
# subsets of an original dataset, with replacement, and fits either a classifier (for classification) or regressor (for regression) to each subset. 
# The predictions for each subset are then aggregated through majority vote for classification or averaging for regression, increasing prediction 
# accuracy.
clf1 = RandomForestClassifier()
clf2 = ExtraTreesClassifier()
clf3 = CatBoostClassifier()
baggingClf1 = BaggingClassifier(base_estimator=clf3, n_estimators=12, max_samples=0.8, random_state=22, verbose=22)

# Train a Machine Learning Model
# #################################### Pay Attention for Future Work!!! ####################################
# Since the focus of this post is about the datetime features, we will just train a random forest model here. 
# Note if you want to use other types of models, you may need to scale or normalize your data. Another thing 
# you may want to do is convert the dayofweek into a categorical variable via one-hot encoding. We don’t need 
# to do these things for a tree-based method though.
baggingClf1.fit(X_train,y_train)

# We can then get our predictions with:
y_pred = baggingClf1.predict(X_test)

# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# print() of the classification_report() with two different formats.
print(classification_report(y_test, y_pred, output_dict=True, digits=4))
print(classification_report(y_test, y_pred, labels=[0, 1, 2], digits=4))

# print() of the True and Predicted values confusion_matrix().
print(confusion_matrix(y_test, y_pred))

# confusion_matrix generation and print to the terminal console.
print(confusion_matrix(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print(type(conf_matrix))

# The AUC score is calculated and printed.
# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)
y_pred = lb.transform(y_pred)
print('ROC AUC score:', roc_auc_score(y_test, y_pred, average='macro'))

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = (time.time() - start_time)
print('The time duration of the experiment was: ')
print("--- %s seconds ---" % (timeDuration))

# confusion_matrix generation and and plot() on the terminal's screen.
# plt.rcParams.update({'font.size': 16})
# plot_confusion_matrix(dt_classifier, X_test, y_test, colorbar = False, cmap=plt.cm.Blues)
# plt.show()

cm = conf_matrix
sns.set(font_scale = 1.75)
# x_axis_labels = ["Positive", "Negative"] # labels for x-axis
# y_axis_labels = ["Positive", "Negative"] # labels for y-axis
x_axis_labels = ["0", "1", "2"] # labels for x-axis
y_axis_labels = ["0", "1", "2"] # labels for y-axis
p = sns.heatmap(cm, annot=True,fmt="d",cmap='Blues', cbar=False, annot_kws={"size":30})
p.xaxis.tick_top() # x axis on top
p.xaxis.set_label_position('top')
p.tick_params(length=0)
p.set( xlabel = "Predicted label", ylabel = "True label", xticklabels = x_axis_labels, yticklabels = y_axis_labels)

plt.show()

