import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
# from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# cm = np.array([[325512, 27985, 49531],[6, 27474, 14],[4, 13931, 26]])
cm = np.array([[397341, 5463], [11897, 29782]])

# [310693  19215 73120]
#  [     0  27489      5]
#  [     0  13961      0]

sns.set(font_scale=1.75)
# Labels for Multiclass Classification
# x_axis_labels = ["0", "1", "2"]  # labels for x-axis
# y_axis_labels = ["0", "1", "2"]  # labels for y-axis
# Labels for Binary Classification
x_axis_labels = ["0", "1"]  # labels for x-axis
y_axis_labels = ["0", "1"]  # labels for y-axis
p = sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 30}
)
p.xaxis.tick_top()  # x-axis on top
p.xaxis.set_label_position("top")
p.tick_params(length=0)
p.set(
    xlabel="Predicted label",
    ylabel="True label",
    xticklabels=x_axis_labels,
    yticklabels=y_axis_labels,
)

plt.show()