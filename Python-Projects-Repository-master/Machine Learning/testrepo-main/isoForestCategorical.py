##############################################################################################
#################### IsolationForest Categorical Unsupervised ML Tutorial ####################
##############################################################################################
# To start, import the following libraries:

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, auc)
import pandas as pd
import sklearn

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
path = r"/Users/christossmiliotopoulos/Desktop/testIFLogfile.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# In the proceeding tutorial, we’ll be working with the breast cancer dataset from the UCI machine learning repository. 
# Fortunately, the scitkit-learn library provides a wrapper function for downloading the data.

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
breast_cancer = load_breast_cancer()
df = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names)
df["benign"] = breast_cancer.target

# As we can see, the dataset contains 30 numerical features and a target value of 0 and 1 for benign and malignant tumors, respectively.
print(df.head())
print(df.tail())

# For our use case, we will assume that a malignant label is anomalous. The dataset contains a relatively high number of malignant tumors. 
# Thus, we make use of downsampling.
majority_df = df[df["benign"] == 1]
minority_df = df[df["benign"] == 0]
minority_downsampled_df = resample(minority_df, replace=True, n_samples=30, random_state=42)
downsampled_df = pd.concat([majority_df, minority_downsampled_df])
print(downsampled_df.head())

# After downsampling, there are over 10x more samples of the majority class than the minority class.
print(downsampled_df["benign"].value_counts())

# We save the features and target as separate variables.
y = downsampled_df["benign"]
X = downsampled_df.drop("benign", axis=1)

# We set a portion of the total data aside for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Next, we create an instance of the IsolationForest class.
model = IsolationForest(random_state=42)

# We train the model.
model.fit(X_train, y_train)

# We predict the data in the test set.
y_pred = model.predict(X_test)
print(y_pred)

# The IsolationForest assigns a value of -1 instead of 0. Therefore, we replace it to ensure we only have 2 distinct values in our confusion matrix.
y_pred[y_pred == -1] = 0
# y_pred = y_pred.replace(['old value'], 'new value')
print(y)

# As we can see, the algorithm does a good job of predicting what data points are anomalous.
confmatrix = confusion_matrix(y_test, y_pred)
print(confmatrix)

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the Multiclass experiment was: ")
print("--- %s seconds ---" % (timeDuration))