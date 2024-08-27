# Naive Bayes is the most straightforward and fast classification algorithm, which is suitable for a large chunk of
# data. Naive Bayes classifier is successfully used in various applications such as spam filtering,
# text classification, sentiment analysis, and recommender systems. It uses Bayes theorem of probability for
# prediction of unknown class.
# Can handle both numeric and categorical data.
# Naive Bayes is a classifier and will therefore perform better with categorical data. Although numeric data will also
# suffice, it assumes all numeric data are normally distributed which is unlikely in real world data.

# Importing Libraries
import matplotlib
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inline
from sklearn import metrics
from sklearn.metrics import accuracy_score, plot_roc_curve, roc_curve, plot_confusion_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from pycm import ConfusionMatrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import time

# Python statement to handle the pandas chained_assignment warning exception.
pd.options.mode.chained_assignment = None  # default='warn'

# Importing the Dataset.
# Read data from CSV file into pandas dataframe.
classificationDataframe = pd.read_csv((
    r"D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Datasets\DATASET\DATASET\AWID-CLS-F-Tst\awid2Tst(all-without-scaling).csv"), sep = ',', error_bad_lines = False, encoding='ISO-8859-1', low_memory=False, verbose=True)


# The purpose of the python functions that follow is to iterate over a dataframe for null, NaN, empty values that affect
print("isnull() check for newAwid2df dataframe:")
print(classificationDataframe.isnull().any())
# classificationDataframe.dropna() will remove the null, NaN and empty values from the dataframe.
classificationDataframe = classificationDataframe.dropna()
print("isnull() check for newAwid2df dataframe after dropna():")
print(classificationDataframe.isnull().any())
print(classificationDataframe['frame.len'].dtypes)
classificationDataframe['frame.len'] = pd.to_numeric(classificationDataframe['frame.len'], errors='coerce')
classificationDataframe = classificationDataframe.dropna()
classificationDataframe = classificationDataframe.fillna(0)
print("isnull() check for newAwid2df dataframe after dropna():")
print(classificationDataframe.isnull().any())

# See the rows and columns of the dataset.
classificationDataframe.shape

# Depict how our dataset actually looks
classificationDataframe.head()

# ======================================================================================================================
# The code up to line "classificationDataframe['class'] = classificationDataframe['class'].astype(int)" is dedicated to
# the preprocessing of AWID2 Reduced [Traning - Test] Datasets. When processing AWID3 Datasets the aforementioned lines
# should be commented due to inconsistency with feature labels. The preprocessing was made column by column, to avoid
# errors with csv manipulation (unwanted null values, NaN) caused by pandas and sklearn.
# ======================================================================================================================

print("======pd.to_numeric() function to transform any non numerical values to numerical======")

classificationDataframe['frame.len'] = pd.to_numeric(classificationDataframe['frame.len'], errors='coerce')
classificationDataframe['radiotap.length'] = pd.to_numeric(classificationDataframe['radiotap.length'], errors='coerce')
classificationDataframe['radiotap.present.tsft'] = pd.to_numeric(classificationDataframe['radiotap.present.tsft'], errors='coerce')
classificationDataframe['radiotap.channel.freq'] = pd.to_numeric(classificationDataframe['radiotap.channel.freq'], errors='coerce')
classificationDataframe['radiotap.channel.type.cck'] = pd.to_numeric(classificationDataframe['radiotap.channel.type.cck'], errors='coerce')
classificationDataframe['radiotap.channel.type.ofdm'] = pd.to_numeric(classificationDataframe['radiotap.channel.type.ofdm'], errors='coerce')
classificationDataframe['radiotap.dbm_antsignal'] = pd.to_numeric(classificationDataframe['radiotap.dbm_antsignal'], errors='coerce')
classificationDataframe['wlan.fc.type'] = pd.to_numeric(classificationDataframe['wlan.fc.type'], errors='coerce')
classificationDataframe['wlan.fc.subtype'] = pd.to_numeric(classificationDataframe['wlan.fc.subtype'], errors='coerce')
classificationDataframe['wlan.fc.ds'] = pd.to_numeric(classificationDataframe['wlan.fc.ds'], errors='coerce')
classificationDataframe['wlan.fc.frag'] = pd.to_numeric(classificationDataframe['wlan.fc.frag'], errors='coerce')
classificationDataframe['wlan.fc.retry'] = pd.to_numeric(classificationDataframe['wlan.fc.retry'], errors='coerce')
classificationDataframe['wlan.fc.pwrmgt'] = pd.to_numeric(classificationDataframe['wlan.fc.pwrmgt'], errors='coerce')
classificationDataframe['wlan.fc.moredata'] = pd.to_numeric(classificationDataframe['wlan.fc.moredata'], errors='coerce')
classificationDataframe['wlan.fc.protected'] = pd.to_numeric(classificationDataframe['wlan.fc.protected'], errors='coerce')
classificationDataframe['wlan.duration'] = pd.to_numeric(classificationDataframe['wlan.duration'], errors='coerce')
classificationDataframe['class'] = pd.to_numeric(classificationDataframe['class'], errors='coerce')

print("isnull() function to replace the null and NaN values to 0")

classificationDataframe['frame.len'].loc[classificationDataframe['frame.len'].isnull()] = 0
classificationDataframe['radiotap.length'].loc[classificationDataframe['radiotap.length'].isnull()] = 0
classificationDataframe['radiotap.present.tsft'].loc[classificationDataframe['radiotap.present.tsft'].isnull()] = 0
classificationDataframe['radiotap.channel.freq'].loc[classificationDataframe['radiotap.channel.freq'].isnull()] = 0
classificationDataframe['radiotap.channel.type.cck'].loc[classificationDataframe['radiotap.channel.type.cck'].isnull()] = 0
classificationDataframe['radiotap.channel.type.ofdm'].loc[classificationDataframe['radiotap.channel.type.ofdm'].isnull()] = 0
classificationDataframe['radiotap.dbm_antsignal'].loc[classificationDataframe['radiotap.dbm_antsignal'].isnull()] = 0
classificationDataframe['wlan.fc.type'].loc[classificationDataframe['wlan.fc.type'].isnull()] = 0
classificationDataframe['wlan.fc.subtype'].loc[classificationDataframe['wlan.fc.subtype'].isnull()] = 0
classificationDataframe['wlan.fc.ds'].loc[classificationDataframe['wlan.fc.ds'].isnull()] = 0
classificationDataframe['wlan.fc.frag'].loc[classificationDataframe['wlan.fc.frag'].isnull()] = 0
classificationDataframe['wlan.fc.retry'].loc[classificationDataframe['wlan.fc.retry'].isnull()] = 0
classificationDataframe['wlan.fc.pwrmgt'].loc[classificationDataframe['wlan.fc.pwrmgt'].isnull()] = 0
classificationDataframe['wlan.fc.moredata'].loc[classificationDataframe['wlan.fc.moredata'].isnull()] = 0
classificationDataframe['wlan.fc.protected'].loc[classificationDataframe['wlan.fc.protected'].isnull()] = 0
classificationDataframe['wlan.duration'].loc[classificationDataframe['wlan.duration'].isnull()] = 0
classificationDataframe['class'].loc[classificationDataframe['class'].isnull()] = 0
classificationDataframe['class'] = classificationDataframe['class'].astype(int)

print("======Re-Check of the dataframe for any forgotten null values======")
print(classificationDataframe.isnull().any())
print(classificationDataframe.isnull().sum())

# Useful python - pandas statements to extract the preprocessed classificationDataframe to .csv file.
# classificationDataframe.to_csv((
#     r"C:\Users\chrsm\Desktop\AWID3 Project Tasks [14-11-21]\Dataset Subsets\all-without-scale01(noNull - index_false).csv"), index=False)
# classificationDataframe.to_csv((
#     r"C:\Users\chrsm\Desktop\all-without-scale02(noNull - index_true).csv"), index=True)


print("Message: Start of X,y declaration and split.")

X = classificationDataframe[['frame.len', 'radiotap.length', 'radiotap.present.tsft', 'radiotap.channel.freq', 'radiotap.channel.type.cck',
    'radiotap.channel.type.ofdm', 'radiotap.dbm_antsignal', 'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.ds',
    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected',
    'wlan.duration']]

y = classificationDataframe[['class']]

print("======isnull() check of X dataframe======")

print(X.isnull().any())

print("======isnull() check of y dataframe======")

print(y.isnull().any())

print("Message: End of X,y declaration and split.")

print("Message: Preparing the stratified K-Fold cross validation technique with K=10 (n_splits=10).")

# Preparing the stratified K-Fold cross validation technique with K=10 (n_splits=10).
skf = StratifiedKFold(n_splits=10)

print("Divide dataframe into training and test sets.")
# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y)
# # inputs and targets concatenated variables.
print("inputs and targets concatenated variables.")
inputs = np.concatenate((X_train, X_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

print("Create a Gaussian Naive Bayes classifier as nb.")
# Create a Gaussian Naive Bayes classifier as nb.
nb = GaussianNB()

# Stratified K-fold Cross Validation model evaluation is going to be executed in parallel with the implementation of
# the Naive Bayes classification model and the extraction of the various metrics for evaluating the classification final
# results. All the functions for the aforementioned evaluation and classification will be included in a for loop of
# K=10 repeated folds.

# Declaration of necessary variables. Variables fold_no, conf_matrix_result, total_accuracy, report_Dataframe and
# start_time will be used in the " for train, test in skf.split(inputs, targets) " loop that follows.
fold_no = 1
conf_matrix_result = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
total_accuracy = 0
report_Dataframe = pd.DataFrame()
start_time = time.time()

print("for loop includes the 10-fold cross validation together with the Naive Bayes classification model.")

# for loop includes the 10-fold cross validation together with the Naive Bayes classification model.
for train, test in skf.split(inputs, targets):
    X_train_skf, X_test_skf = inputs[train], inputs[test]
    # print(X_train_skf)
    # print(X_test_skf)
    y_train_skf, y_test_skf = targets[train], targets[test]
    # print(y_train_skf)
    # print(y_test_skf)
    nb.fit(X_train_skf, y_train_skf)  # Naive Bayes model nb.fit algorithm training.
    y_pred_skf = nb.predict(X_test_skf)  # y_pred_skf variable with the prediction of the classification model.

    # confusion_matrix generation and print to the terminal console.

    print(confusion_matrix(y_test_skf, y_pred_skf))
    conf_matrix = confusion_matrix(y_test_skf, y_pred_skf)
    conf_matrix_result = conf_matrix_result + conf_matrix
    plot_confusion_matrix(nb, X_test_skf, y_test_skf)
    plt.show()

    # Model Accuracy, how often is the classifier correct per cross validation fold.
    print(f"Fold {str(fold_no)} Accuracy:", metrics.accuracy_score(y_test_skf, y_pred_skf))

    # Calculation of the total accuracy of the Naive Bayes classification model.
    print("The total accuracy of the Naive Bayes classification model is: ")
    total_accuracy = total_accuracy + metrics.accuracy_score(y_test_skf, y_pred_skf)
    print(total_accuracy/fold_no)

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

# ======================================================================================================================
# ======================================================================================================================
# The rest of the code is still under construction and review regarding its usefulness.
# ======================================================================================================================
# ======================================================================================================================

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
