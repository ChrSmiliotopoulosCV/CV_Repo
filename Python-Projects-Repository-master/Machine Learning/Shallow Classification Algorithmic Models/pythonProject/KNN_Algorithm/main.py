# Package Imports
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt

# Read in the data using pandas
df = pd.read_csv(
    r'D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\KNN Algorithm\k-NN-Tutorial-Scikit-learn-master\data\diabetes_data.csv')

# Check data has been read in properly
df.head()

# Check number of rows and columns in dataset
print("The number of rows and columns in the dataset is:", df.shape)

# Create a dataframe with all training data except the target column
X = df.drop(columns=['diabetes'])
X.shape

# check that the target variable has been removed
X.head()

# separate target values
y = df['diabetes'].values

# view target values
print(y[0:5])

# split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the data
knn.fit(X_train, y_train)

# show first 5 model predictions on the test data
# knn_predict = knn.predict(X_test)[0:5]
knn_predict = knn.predict(X_test)

# check accuracy of our model on the test data
knn_score = knn.score(X_test, y_test)
print(knn_score)

# create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)

# train model with cv of 5
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

# print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

# create new a knn model
knn2 = KNeighborsClassifier()

# create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}

# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

# fit model to data
knn_gscv.fit(X, y)

# check top performing n_neighbors value

var_top_performing = knn_gscv.best_params_
print(var_top_performing)

# check mean score for the top performing value of n_neighbors

mean_top_performing = knn_gscv.best_score_
label_01 = "Mean of top_performing:"
print(label_01, mean_top_performing)

# Accuracy Score - Implementation of Scikit-Learn's function, accuracy_score, which accepts the true value and
# predicted value as its input to calculate the accuracy score of a model.

knn_score2 = accuracy_score(y_test, knn_predict)
print("Accuracy score (KNN): ", knn_score2)

# Evaluate the performance of KNN classifier with the use of confusion matrix.
print("Confusion_matrix:")
print(confusion_matrix(y_test, knn_predict))

tn, fp, fn, tp = confusion_matrix(y_test, knn_predict).ravel()

conf_matrix = confusion_matrix(y_test, knn_predict)

print(classification_report(y_test, knn_predict))

# As of scikit-learn v0.20, the easiest way to convert a classification report to a pandas Dataframe is by simply
# having the report returned as a dict.
report = classification_report(y_test, knn_predict, output_dict=True)

# Construct a Dataframe and transpose it
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = knn_score
df_report['cv_scores'] = cv_scores
df_report['mean_top_performing'] = mean_top_performing

# Extract fn, fp, tn, tp from confusion matrix and add as columns to the dataframe.
df_report['FN'] = fn
df_report['FP'] = fp
df_report['TN'] = tn
df_report['TP'] = tp

#print(str(report['0']['recall']))

# Measure and compare our classifiers’ performance by calculating the area under the curve (AUC) which will
# result in a score called AUROC (Area Under the Receiver Operating Characteristics). A perfect AUROC should
# have a score of 1 whereas a random classifier will have a score of 0.5.
fpr, tpr, thresholds = metrics.roc_curve(y_test, knn_predict, pos_label=0)
plot_roc_curve(knn, X_test, y_test)
print(plot_roc_curve(knn, X_test, y_test))
plt.show()

# AUC metric calculation Method_1
auc = np.trapz(fpr, tpr)
print(auc)
df_report['AUC'] = auc

# AUC metric calculation Method_2s
auc_score1 = roc_auc_score(y_test, knn_predict)
print(auc_score1)