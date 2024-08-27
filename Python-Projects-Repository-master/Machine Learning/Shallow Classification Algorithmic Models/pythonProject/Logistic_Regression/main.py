# Logistic Regression on IRIS dataset
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score

# Importing the Dataset
# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Ρead data from CSV file into pandas dataframe
dataset = pd.read_csv((
    "D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\SVM Algorithm\iris.csv"),
    names=colnames)
dataset_statistics = dataset.describe()

# Splitting the dataset into the Training set and Test set
X = dataset.iloc[:, [0, 1, 2, 3]].values
y = dataset.iloc[:, 4].values

# Feature Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Predict probabilities
probs_y = classifier.predict_proba(X_test)

# # Print results
# probs_y = np.round(probs_y, 2)
# res = "{:<10} | {:<10} | {:<10} | {:<13} | {:<5}".format("y_test", "y_pred", "Setosa(%)", "versicolor(%)",
#                                                          "virginica(%)\n")
# res += "-" * 65 + "\n"
# res += "\n".join("{:<10} | {:<10} | {:<10} | {:<13} | {:<10}".format(x, y, a, b, c) for x, y, a, b, c in
#                  zip(y_test, y_pred, probs_y[:, 0], probs_y[:, 1], probs_y[:, 2]))
# res += "\n" + "-" * 65 + "\n"
# print(res)

# Making the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)


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

# Accuracy Score - Implementation of Scikit-Learn's function, accuracy_score, which accepts the true value and
# predicted value as its input to calculate the accuracy score of a model.

svm_score = accuracy_score(y_test, y_pred)
print("Accuracy score (SVM): ", svm_score)

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(y_test, y_pred, output_dict=True)

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = svm_score
df_report['AUC'] = auc_score
