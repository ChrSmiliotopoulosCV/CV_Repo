import sys

import matplotlib
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import pyplot
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_curve, plot_confusion_matrix, auc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn import inline

# Python statement to handle the pandas chained_assignment warning exception.
pd.options.mode.chained_assignment = None  # default='warn'


# class Logger(object) will write the results of ML classification algorithms analysis to both stdout and a logfile.
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfileRandomForest.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


# This is the actual argument with which sys.stdout will be printed to both stdout and a logfile.
sys.stdout = Logger()

# Importing the Dataset.
# Read data from CSV file into pandas dataframe.
voting_Dataframe = pd.read_csv((
    r"C:\Users\chrsm\Desktop\awid2-new\1-test-16-ready-csv.csv"), low_memory=False, verbose=True)

# See the rows and columns of the dataset.
voting_Dataframe.shape

# Depict how our dataset actually looks
voting_Dataframe.head()

# The purpose of the python functions that follow is to iterate over a dataframe for null, NaN, empty values that
# affect the performance and execution of the algorithm.
print(voting_Dataframe.isnull().any())

# Divide the Label column from the rest of the dataframe.
X = voting_Dataframe.drop('Label', axis=1)
y = voting_Dataframe['Label']

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y)

# # Create a Gaussian Naive Bayes classifier as nb.
nb = GaussianNB()

# # Decision Tree Classifier
# # The general motive of using a Decision Tree is to create a training model which can be used to predict the class or
# # value of target variables by learning decision rules inferred from prior data(training data).
classifier = DecisionTreeClassifier(random_state=0, criterion="entropy", splitter="random", max_depth=2)

# Random Forest Classifier Random Forest is a tree-based machine learning method used for classification, regression,
# and clustering as well. It uses a collection of Trees, where each tree votes on the outcome. In the case of
# classification, the forest gives the class voted by the maximum number of trees. In the case of regression,
# the output is the average of the outcomes of all the trees. In classification with Random Forest algorithm,
# each tree is given a sample of the full data set with replacement. A subset of features is used to create the tree
# to the maximum depth possible. The data points and features are then used to create the respective CART trees.
# Instantiate Random Forest classification model with 1000 decision trees.
# rf_classifier = RandomForestClassifier(n_estimators=350, random_state=42, criterion="entropy",
#                                        max_depth=3, max_features=0.4,
#                                        min_samples_leaf=3, verbose=2, n_jobs=-1)

rf_classifier = RandomForestClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=150, n_estimators=350,
                                       max_samples=0.2, max_features=6, n_jobs=-1, verbose=2)
#
# rf_classifier = RandomForestClassifier(max_depth=4, min_samples_split=20, min_samples_leaf=150, n_estimators=350,
#                                        max_samples=0.2, max_features=6, max_leaf_nodes=10, n_jobs=-1, verbose=2)
