# K-means Clustering:
# K-Means Clustering is an unsupervised machine learning algorithm. In contrast to traditional supervised machine
# learning algorithms, K-Means attempts to classify data without having first been trained with labeled data. Once the
# algorithm has been run and the groups are defined, any new data can be easily assigned to the most relevant group.
# The goal of the K-means clustering algorithm is to find groups in the data, with the number of groups represented
# by the variable K. The algorithm works iteratively to assign each data point to one of the K groups based on the
# features that are provided.
# The outputs of executing a K-means on a dataset are:
# K centroids: Centroids for each of the K clusters identified from the dataset.
# Labels for the training data: Complete dataset labelled to ensure each data point is assigned to one of the clusters.

# Importing Libraries
import inline
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

sns.set_style('whitegrid')
# %matplotlib inline
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, plot_roc_curve
from sklearn.metrics import accuracy_score

# Importing Data
df = pd.read_csv(
    'D:\Personal Projects\Διδακτορικό\AWID3 Machine Learning Project\Feature Selection and ML\Classification Algorithms Preparation Folder\K-Means Algorithm\College_Data.csv',
    index_col=0)
# Check the head of the data
df.head()
# Check the info() and describe() methods on the data.
df.info()
# Create a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column.
sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, fit_reg=False, hue='Private', height=6, aspect=1)
plt.show()
# Create a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.
sns.lmplot(x='Outstate', y='F.Undergrad', data=df, fit_reg=False, hue='Private', height=6, aspect=1)
plt.show()
# Create a stacked histogram showing Out of State Tuition based on the Private column.
g = sns.FacetGrid(df, hue='Private', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.7)
plt.show()
# Create a similar histogram for the Grad.Rate column.
g = sns.FacetGrid(df, hue='Private', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.7)
plt.show()
# Notice how there seems to be a private school with a graduation rate of higher than 100%.What is the name of that
# school?
highGradeSchool = df[df['Grad.Rate'] > 100]
print(highGradeSchool)

# K Means Cluster Creation
# Create an instance of a K Means model with 2 clusters.
myKMC = KMeans(n_clusters=2)
# Fit the model to all the data except for the Private label.
myKMC.fit(df.drop('Private', axis=1))
# What are the cluster center vectors?
myKMC.cluster_centers_
clusterCenters = myKMC.cluster_centers_
print(clusterCenters)
# Create a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.
df['Cluster'] = df['Private'].apply(lambda x: 1 if x == 'Yes' else 0)
df.head()
# Create a confusion matrix and classification report to see how well the Kmeans clustering worked without being
# given any labels.
print(confusion_matrix(df['Cluster'], myKMC.labels_))
tn, fp, fn, tp = confusion_matrix(df['Cluster'], myKMC.labels_).ravel()
conf_matrix = confusion_matrix(df['Cluster'], myKMC.labels_)

# Evaluate the performance of K-Means classifier with the use of classification_report() function.
print(classification_report(df['Cluster'], myKMC.labels_))
report = classification_report(df['Cluster'], myKMC.labels_)

# Accuracy Score - Implementation of Scikit-Learn's function, accuracy_score, which accepts the true value and
# predicted value as its input to calculate the accuracy score of a model.
acc = accuracy_score(df['Cluster'], myKMC.labels_)
print("Accuracy score is", acc)

# Construct a Dataframe and transpose it
report = classification_report(df['Cluster'], myKMC.labels_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report['model accuracy'] = acc
# Extract fn, fp, tn, tp from confusion matrix and add as columns to the dataframe.
df_report['FN'] = fn
df_report['FP'] = fp
df_report['TN'] = tn
df_report['TP'] = tp
# Measure and compare our classifiers’ performance by calculating the area under the curve (AUC) which will
# result in a score called AUROC (Area Under the Receiver Operating Characteristics). A perfect AUROC should
# have a score of 1 whereas a random classifier will have a score of 0.5.
fpr, tpr, thresholds = metrics.roc_curve(df['Cluster'], myKMC.labels_, pos_label=0)
auc = np.trapz(fpr, tpr)
print(auc)
df_report['AUC'] = auc