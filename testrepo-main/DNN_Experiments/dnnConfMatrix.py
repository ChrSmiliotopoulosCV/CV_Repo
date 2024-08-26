import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

cm = np.array([[401557, 3, 1805],[101, 27162, 97],[2403, 0, 11355]])


sns.set(font_scale=1.75)
# x_axis_labels = ["Positive", "Negative"] # labels for x-axis
# y_axis_labels = ["Positive", "Negative"] # labels for y-axis
x_axis_labels = ["0", "1", "2"]  # labels for x-axis
y_axis_labels = ["0", "1", "2"]  # labels for y-axis
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
