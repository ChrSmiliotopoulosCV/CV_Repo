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
from sklearn.metrics import (
    accuracy_score,
    plot_roc_curve,
    roc_curve,
    plot_confusion_matrix,
    auc,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
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
        self.file = open(filename, "w")

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


start_time = time.time()

# The path of the destination folder on which the logs.txt file will be created.
# path = (r"C:\Users\chrsm\Desktop\ExtractedFiles [18 Jan 2023]\Swallow_Classification_Results\DecisionTree\2nd Set of Experiments [ExtendedDF]\dtfullLogs.txt")
path = r"C:\Users\chrsm\Desktop\ExtractedFiles [18 Jan 2023]\Swallow_Classification_Results\KNN\2nd Set of Experiments [ExtendedDF]\knnfullLogs1.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        r"C:\Users\chrsm\Desktop\ExtractedFiles [18 Jan 2023]\newExtendedSchema\Preprocessing_Dataframes [Final Extraction]\1.8M_FinallySelected\Best Results\full-csv.csv"
    ),
    encoding="ISO-8859-1",
    low_memory=False,
)
df.head()
print(df)
print(
    "Each Label within the oheMinMaxPreProcessedDataset.csv is comprised from the following elements: "
)
print(df["Label"].value_counts())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
print(df)
X = df.drop("Label", axis=1)
y = df["Label"]

# print(X)
# print(y)

# Decision Tree Classifier
# The general motive of using a Decision Tree is to create a training model which can be used to predict the class or
# value of target variables by learning decision rules inferred from prior data(training data).
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=30)
dt_classifier1 = DecisionTreeClassifier(
    random_state=0, criterion="entropy", splitter="random", max_depth=2
)
dt_classifier2 = DecisionTreeClassifier(
    random_state=0,
    criterion="entropy",
    splitter="random",
    max_depth=2,
    max_leaf_nodes=100,
    min_samples_leaf=2,
    ccp_alpha=0.0001,
)

# Random Forest Classifier Random Forest is a tree-based machine learning method used for classification, regression,
# and clustering as well. It uses a collection of Trees, where each tree votes on the outcome. In the case of
# classification, the forest gives the class voted by the maximum number of trees. In the case of regression,
# the output is the average of the outcomes of all the trees. In classification with Random Forest algorithm,
# each tree is given a sample of the full data set with replacement. A subset of features is used to create the tree
# to the maximum depth possible. The data points and features are then used to create the respective CART trees.
# Instantiate Random Forest classification model with 1000 decision trees.

# This was the first successful initialization of the rf_classifier with n_estimators=350 and max_depth=3/
# rf_classifier = RandomForestClassifier(n_estimators=350, random_state=42, criterion="entropy",
#                                        max_depth=3, max_features=0.4,
#                                        min_samples_leaf=3, verbose=2, n_jobs=-1)
rf_classifier = RandomForestClassifier(verbose=2)

# This was the final successful initialization of the rf_classifier, that was finally used to execute the experiment with n_estimators=1000 and max_depth=4.
rf_classifier1 = RandomForestClassifier(
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=150,
    n_estimators=1000,
    max_samples=0.2,
    max_features=6,
    n_jobs=-1,
    verbose=2,
)

# This rf_classifier implementation didn't succeed to avoid algorithm's overfitting with n_estimators=350 and max_depth=4.
rf_classifier2 = RandomForestClassifier(
    max_depth=4,
    min_samples_split=20,
    min_samples_leaf=150,
    n_estimators=350,
    max_samples=0.2,
    max_features=6,
    max_leaf_nodes=10,
    n_jobs=-1,
    verbose=2,
)

# LightGBM is a fast, distributed, high performance gradient boosting framework based on decision tree algorithms, used for ranking, classification and many
# other machine learning tasks.
lgbmclassifier1 = lgb.LGBMClassifier(
    max_depth=12,
    num_leaves=4000,
    min_data_in_leaf=6500,
    learning_rate=0.99,
    verbose=5,
    lambda_l1=15,
    lambda_l2=15,
    min_gain_to_split=0.12,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.5,
)

lgbmclassifier2 = lgb.LGBMClassifier(
    max_depth=6,
    num_leaves=1700,
    min_data_in_leaf=7500,
    learning_rate=0.15,
    verbose=5,
    lambda_l1=0.03,
    lambda_l2=0,
    min_gain_to_split=12.71,
    bagging_fraction=0.3,
    bagging_freq=1,
    feature_fraction=0.7,
)

lgbmclassifier3 = lgb.LGBMClassifier(
    learning_rate=0.1,
    max_bin=20,
    max_depth=20,
    min_child_samples=30,
    min_data_in_bin=10,
    min_split_gain=0.1,
    n_estimators=100,
    num_leaves=20,
    reg_alpha=0.01,
    reg_lambda=0.01,
    n_jobs=1,
    verbose=5,
)

# Similar to Random Forests, ExtraTrees is an ensemble ML approach that trains numerous decision trees and aggregates the results from the group of decision
# trees to output a prediction.

extraTreesClf1 = ExtraTreesClassifier(n_estimators=300, random_state=42)

extraTreesClf2 = ExtraTreesClassifier(
    n_estimators=100,
    max_leaf_nodes=100,
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=0,
)

extraTreesClf3 = ExtraTreesClassifier(
    max_depth=300,
    n_estimators=100,
    max_leaf_nodes=100,
    min_samples_leaf=2,
    min_samples_split=10,
    ccp_alpha=0.0001,
    random_state=0,
)

# Stochastic Gradient Descent (SGD) is a simple yet efficient optimization algorithm used to find the values of parameters/coefficients of functions that minimize a cost function.
# In other words, it is used for discriminative learning of linear classifiers under convex loss functions such as SVM and Logistic regression. It has been successfully applied to
# large-scale datasets because the update to the coefficients is performed for each training instance, rather than at the end of instances.
sgdc1 = SGDClassifier(
    max_iter=5000,
    tol=0.00000001,
    loss="modified_huber",
    early_stopping=True,
    learning_rate="optimal",
    class_weight="balanced",
    verbose=5,
)
sgdc2 = SGDClassifier(
    learning_rate="optimal",
    loss="log",
    max_iter=3000000,
    validation_fraction=0.5,
    n_jobs=-1,
    verbose=5,
    early_stopping=True,
)

# The Linear Support Vector Classifier (SVC) method applies a linear kernel function to perform classification and it performs well with a large number of samples. If we compare it
# with the SVC model, the Linear SVC has additional parameters such as penalty normalization which applies 'L1' or 'L2' and loss function. The kernel method can not be changed in
# linear SVC, because it is based on the kernel linear method.
linearSVClf1 = LinearSVC(max_iter=30000, C=1.5, verbose=5)
linearSVClf2 = LinearSVC(verbose=5)

# Logistic Regression is a statistical approach and a Machine Learning algorithm that is used for classification problems and is based on the concept of probability. In the multiclass
# case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to
# ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ solvers.)
logiRegClf1 = LogisticRegression(
    solver="sag", max_iter=1000, tol=0.01, multi_class="multinomial", verbose=5
)
logiRegClf2 = LogisticRegression(verbose=5)

# CatBoost is a high-performance open source library for gradient boosting on decision trees.
catBoostClf1 = CatBoostClassifier(
    iterations=70,
    learning_rate=0.3,
    verbose=True,
    l2_leaf_reg=0.0000001
    # loss_function='CrossEntropy'
)

# K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification. KNN
# tries to predict the correct class for the test data by calculating the distance between the test data and all the
# training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the
# probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will
# be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.
# knnJaccardClf1 = KNeighborsClassifier(n_neighbors=3, metric="jaccard")
# knnJaccardClf1 = KNeighborsClassifier(weights='distance', algorithm='brute', p=1, n_jobs=-1, metric='euclidean')
knnJaccardClf1 = KNeighborsClassifier(
    algorithm="auto",
    leaf_size=30,
    metric="minkowski",
    metric_params=None,
    n_jobs=None,
    n_neighbors=5,
    p=2,
    weights="uniform",
)

# Gaussian Naive Bayes supports continuous valued features and models each as conforming to a Gaussian (normal) distribution.
# An approach to create a simple model is to assume that the data is described by a Gaussian distribution with no co-variance
# (independent dimensions) between dimensions.
nbclf1 = GaussianNB()

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

# Train a Machine Learning Model
# #################################### Pay Attention for Future Work!!! ####################################
# Since the focus of this post is about the datetime features, we will just train a random forest model here.
# Note if you want to use other types of models, you may need to scale or normalize your data. Another thing
# you may want to do is convert the dayofweek into a categorical variable via one-hot encoding. We don’t need
# to do these things for a tree-based method though.
knnJaccardClf1.fit(X_train, y_train)

# We can then get our predictions with:
y_pred = knnJaccardClf1.predict(X_test)

# The Accuracy of the sklearn metrics is printed on the terminal's screen.
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

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
print("ROC AUC score:", roc_auc_score(y_test, y_pred, average="macro"))

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the experiment was: ")
print("--- %s seconds ---" % (timeDuration))

# confusion_matrix generation and and plot() on the terminal's screen.
# plt.rcParams.update({'font.size': 16})
# plot_confusion_matrix(dt_classifier, X_test, y_test, colorbar = False, cmap=plt.cm.Blues)
# plt.show()

# plot_confusion_matrix(dt_classifier, X_test, y_test, values_format='.2f')
# plt.savefig('ML2[05-04-2022]/conf_matrix-fold_no' + str(fold_no) + '.png', bbox_inches="tight")
# plt.show()

cm = conf_matrix
sns.set(font_scale=1.75)
# x_axis_labels = ["Positive", "Negative"] # labels for x-axis
# y_axis_labels = ["Positive", "Negative"] # labels for y-axis
x_axis_labels = ["0", "1", "2"]  # labels for x-axis
y_axis_labels = ["0", "1", "2"]  # labels for y-axis
p = sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 30}
)
p.xaxis.tick_top()  # x axis on top
p.xaxis.set_label_position("top")
p.tick_params(length=0)
p.set(
    xlabel="Predicted label",
    ylabel="True label",
    xticklabels=x_axis_labels,
    yticklabels=y_axis_labels,
)

plt.show()
