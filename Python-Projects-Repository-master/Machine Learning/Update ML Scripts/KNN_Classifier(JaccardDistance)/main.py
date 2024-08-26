# Importing Libraries
import matplotlib
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inline
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time

# Python statement to handle the pandas chained_assignment warning exception.
pd.options.mode.chained_assignment = None  # default='warn'

# Importing the Dataset.
# Read data from CSV file into pandas dataframe.
knn_Dataframe = pd.read_csv((
    r"C:\Users\chrsm\Desktop\AWID3 Project Tasks [14-11-21]\Dataset Subsets\Trial Datasets\enchanced-reduced.csv"),
    low_memory=False)

# See the rows and columns of the dataset.
knn_Dataframe.shape

# Depict how our dataset actually looks
knn_Dataframe.head()

# # The purpose of the python functions that follow is to iterate over a dataframe for null, NaN, empty values that affect
# print(knn_Dataframe.isnull().any())
# # print(knn_Dataframe.index[np.isinf(knn_Dataframe).any(1)])
# print(knn_Dataframe['Label'].isnull().values.any())
# print(knn_Dataframe['Label'].isnull().sum())
# new = knn_Dataframe['Label'].loc[knn_Dataframe['Label'].isnull()]
# print(new)
# knn_Dataframe['Label'].loc[knn_Dataframe['Label'].isnull()] = 0
# print(knn_Dataframe['Label'].isnull().sum())
# new = knn_Dataframe['Label'].loc[knn_Dataframe['Label'].isnull()]
# print(new)
# knn_Dataframe.to_csv((
#     r"C:\Users\chrsm\Desktop\AWID3 Project Tasks [14-11-21]\Dataset Subsets\all-without-scale01(noNull - index_false).csv"), index=False)
# knn_Dataframe.to_csv((
#     r"C:\Users\chrsm\Desktop\AWID3 Project Tasks [14-11-21]\Dataset Subsets\all-without-scale02(noNull - index_true).csv"), index=True)
# print(new)
# knn_Dataframe = knn_Dataframe.fillna(lambda x: x.median())
# print(knn_Dataframe.isnull().any())

# Divide the Label column from the rest of the dataframe.
X = knn_Dataframe.drop('Label', axis=1)
y = knn_Dataframe['Label']

# Preparing the stratified K-Fold cross validation technique with K=10 (n_splits=10).
skf = StratifiedKFold(n_splits=10)

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y)
# inputs and targets concatenated variables.
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# # Create a Gaussian Naive Bayes classifier as nb.
# nb = GaussianNB()

# # Decision Tree Classifier
# # The general motive of using a Decision Tree is to create a training model which can be used to predict the class or
# # value of target variables by learning decision rules inferred from prior data(training data).
# classifier = DecisionTreeClassifier(random_state=0, criterion="entropy", splitter="random", max_depth=2)

# Random Forest Classifier Random Forest is a tree-based machine learning method used for classification, regression,
# and clustering as well. It uses a collection of Trees, where each tree votes on the outcome. In the case of
# classification, the forest gives the class voted by the maximum number of trees. In the case of regression,
# the output is the average of the outcomes of all the trees. In classification with Random Forest algorithm,
# each tree is given a sample of the full data set with replacement. A subset of features is used to create the tree
# to the maximum depth possible. The data points and features are then used to create the respective CART trees.
# Instantiate Random Forest classification model with 1000 decision trees.
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# An AdaBoost [1] classifier is a meta-estimator that begins by fitting a classifier on the original dataset and then
# fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified
# instances are adjusted such that subsequent classifiers focus more on difficult cases. This class implements the
# algorithm known as AdaBoost-SAMME [2].
# adaBoost_Classifier = AdaBoostClassifier(n_estimators=100, random_state=0)

# Create KNN classifier
knnClassifier = KNeighborsClassifier(n_neighbors=3, metric="jaccard")

# Stratified K-fold Cross Validation model evaluation is going to be executed in parallel with the implementation of
# the Naive Bayes classification model and the extraction of the various metrics for evaluating the classification final
# results. All the functions for the aforementioned evaluation and classification will be included in a for loop of
# K=10 repeated folds.

# Declaration of necessary variables. Variables fold_no, conf_matrix_resuld, total_accuracy, report_Dataframe and
# start_time will be used in the " for train, test in skf.split(inputs, targets) " loop that follows.
fold_no = 1
conf_matrix_result = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
total_accuracy = 0
report_Dataframe = pd.DataFrame()
start_time = time.time()

# for loop includes the 10-fold cross validation together with the Naive Bayes classification model.
for train, test in skf.split(inputs, targets):
    X_train_skf, X_test_skf = inputs[train], inputs[test]
    # print(X_train_skf)
    # print(X_test_skf)
    y_train_skf, y_test_skf = targets[train], targets[test]
    # print(y_train_skf)
    # print(y_test_skf)
    knnClassifier.fit(X_train_skf, y_train_skf)  # Naive Bayes model nb.fit algorithm training.
    y_pred_skf = knnClassifier.predict(X_test_skf)  # y_pred_skf variable is used to store in memory the prediction of
    # the classification model.

    # confusion_matrix generation and print to the terminal console.

    print(confusion_matrix(y_test_skf, y_pred_skf))
    conf_matrix = confusion_matrix(y_test_skf, y_pred_skf)
    conf_matrix_result = conf_matrix_result + conf_matrix
    plot_confusion_matrix(knnClassifier, X_test_skf, y_test_skf)
    plt.show()

    # Model Accuracy, how often is the classifier correct per cross validation fold.
    print(f"Fold {str(fold_no)} Accuracy:", metrics.accuracy_score(y_test_skf, y_pred_skf))

    # Calculation of the total accuracy of the Naive Bayes classification model.
    print("The total accuracy of the Naive Bayes classification model is: ")
    total_accuracy = total_accuracy + metrics.accuracy_score(y_test_skf, y_pred_skf)
    print(total_accuracy / fold_no)

    # Evaluate the performance of Naive Bayes classifier with the use of classification_report() function.
    print(f"Fold {str(fold_no)} Classification Report: \n", classification_report(y_test_skf, y_pred_skf))

    # As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
    # having the report returned as a dict.
    report = classification_report(y_test_skf, y_pred_skf, output_dict=True)

    # Construct a Dataframe and transpose it, that stores the report variable.
    df_report = pd.DataFrame(report).transpose()

    # report_Dataframe .add function to add the 10 classification reports produced by the 10-fold cross validation.
    report_Dataframe = report_Dataframe.add(df_report, fill_value=0)

    # The duration of the execution of each fold in seconds.
    print((f"The duration of Fold {str(fold_no)} was:"))
    print("--- %s seconds ---" % (time.time() - start_time))

    # Increase fold number.
    fold_no = fold_no + 1

# print() function for the mean values of conf_matrix_result, total_accuracy, report_Dataframe.
print("The mean value of confusion matrices after the 10-Fold Cross Validation has as follows:")
print(conf_matrix_result // 10)
print("The mean value of the accuracy from each fold of 10-fold CV has as follows:")
print(total_accuracy / 10)
print("The mean value of classification reports from each fold of 10-fold CV has as follows:")
print(report_Dataframe.round(1) / 10)

# # False/True Positive Rate, thresholds, auc calculation with the metrics.roc_curve function.
# fpr, tpr, thresholds = metrics.roc_curve(y_test_skf, y_pred_skf, pos_label=0)
# auc = np.trapz(fpr, tpr)
# print("The ROC-AUC value for the classification model has as follows:")
# print(auc)


# # LabelBinarizer() function in order to evaluate the AUC ROC score for our multi-class problem
# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#     return roc_auc_score(y_test, y_pred, average=average)
#
#
# # auc_score calculation and print()
# auc_score = multiclass_roc_auc_score(y_test_skf, y_pred_skf)
# print(auc_score)

# The rest of the code has been used in classification algorithms execution examples as an aid to extract
# collectively all the calculated results in one pandas Dataframe.

# # Function to obtain TP, FN FP, and TN for each class in the confusion matrix of our multiclass problem
# def counts_from_confusion(conf_matrix):
#     """
#     Obtain TP, FN FP, and TN for each class in the confusion matrix
#     """
#
#     counts_list = []
#
#     # Iterate through classes and store the counts
#     for i in range(conf_matrix.shape[0]):
#         tp = conf_matrix[i, i]
#
#         fn_mask = np.zeros(conf_matrix.shape)
#         fn_mask[i, :] = 1
#         fn_mask[i, i] = 0
#         fn = np.sum(np.multiply(conf_matrix, fn_mask))
#
#         fp_mask = np.zeros(conf_matrix.shape)
#         fp_mask[:, i] = 1
#         fp_mask[i, i] = 0
#         fp = np.sum(np.multiply(conf_matrix, fp_mask))
#
#         tn_mask = 1 - (fn_mask + fp_mask)
#         tn_mask[i, i] = 0
#         tn = np.sum(np.multiply(conf_matrix, tn_mask))
#
#         counts_list.append({'Class': i,
#                             'TP': tp,
#                             'FN': fn,
#                             'FP': fp,
#                             'TN': tn})
#     return counts_list
#
#
# # TP, FN FP, and TN variable storing and print()
# countlist = counts_from_confusion(conf_matrix)
# print(countlist)
#
# # TP, FN FP, and TN DataFrame() creation and Index[] manipulation
# df_confusion_countlist = pd.DataFrame(countlist)
# df_confusion_countlist.Class[df_confusion_countlist.Class == 0] = 'Flooding'
# df_confusion_countlist.Class[df_confusion_countlist.Class == 1] = 'Impersonation'
# df_confusion_countlist.Class[df_confusion_countlist.Class == 2] = 'Normal'
# # # df_confusion_countlist.Class[df_confusion_countlist.Class == 3] = 'Evil_Twin'
# # # df_confusion_countlist.Class[df_confusion_countlist.Class == 4] = 'Kr00k'
# # # df_confusion_countlist.Class[df_confusion_countlist.Class == 5] = 'Krack'
# # # df_confusion_countlist.Class[df_confusion_countlist.Class == 6] = 'Normal'
# # # df_confusion_countlist.Class[df_confusion_countlist.Class == 7] = 'Rogue_AP'
#
#
# # Evaluate the performance of SVM classifier with the use of classification_report() function.
# print(classification_report(y_test, y_pred))
#
# # LabelBinarizer() function in order to evaluate the AUC ROC score for our multi-class problem
# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#     return roc_auc_score(y_test, y_pred, average=average)
#
# # auc_score calculation and print()
# auc_score = multiclass_roc_auc_score(y_test, y_pred)
# print(auc_score)
#
# # As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# # having the report returned as a dict.
# report = classification_report(y_test, y_pred, output_dict=True)
#
# # Construct a Dataframe and transpose it
# df_report = pd.DataFrame(report).transpose()
# df_report['model accuracy'] = accuracyScore
# df_report['AUC'] = auc_score
