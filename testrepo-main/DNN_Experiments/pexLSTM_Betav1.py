import time
import sys

import pandas as pd
import numpy as np
import seaborn as sns
from keras.layers import LSTM, Dense
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split

import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


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


pd.options.display.max_rows = 999
start_time = time.time()

# The path of the destination folder on which the logs.txt file will be created.
# path = r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/pexMLPlogfile(UML)007LSTM.txt"
path = r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\DNN_Experiments\pexMLPlogfile(UML)008LSTM_Finalfullset.txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)
print(pd.__version__)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        # r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/binary-csv-csv[BinaryFeaturesReduced].csv"
        r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\binary-csv-csv[BinaryFeaturesReduced].csv"
        # r"/Users/christossmiliotopoulos/Documents/GitHub/testrepo/LMD-2023 Dataset/full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
        # r"C:\Users\chrsm\Desktop\Paper_no4 - Unsupervised ML LMD-2023 Paper\GitHub_Repository\testrepo\LMD-2023 Dataset\full-csv(Evaluated-TitlesFeaturesReduced)-csv.csv"
    ),
    encoding="ISO-8859-1",
    low_memory=False,
)
df.head()
print(df)
print(
    "Each Label within the oheMinMaxPreProcessedDataset.csv (full-csv(Evaluated Titles).csv) is comprised from the "
    "following elements:"
)
print(df["Label"].value_counts())
print(df.isnull().sum())

# shuffle the DataFrame rows and divide the Label column from the rest of the df dataframe.
df = df.sample(frac=1)
# print(df.info())
print(df.head())
X = df.drop("Label", axis=1)
y = df["Label"]

# Xnew = df[df['Label'] == 0]
# print(Xnew.head())
# print(Xnew["Label"].value_counts())

# # Keras requires your output feature to be one-hot encoded values.
# Y_final = tf.keras.utils.to_categorical(y)
# print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))
# print(Y_final)

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.005, test_size=0.0025)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

# Keep only the normal data for the training dataset
X_train = X_train.where(y_train == 0)
print(X_train.head())
print("The length of normal X_train is ", len(X_train))
X_train = X_train.dropna()
print(X_train.head())
print("The length of normal dropna() X_train is ", len(X_train))

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
print("The shape of the reshaped train subsets is: ")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train.shape[1])
print(X_train.shape[2])

# Search the len() of the df and X dataframes to be used as input in the input_dim() variable of Keras.
dfColumns = len(df.columns)
XColumns = len(X.columns)
print("--- The length of the df.columns is %s --" % dfColumns)
print("--- The length of the X.columns is %s --" % XColumns)

# Build the Sequential Tensorflow Keras Multilayer Perceptron (MLP) model.
input_dim = len(df.columns) - 1
print("--- The input_dim() aka the number of input features is %s ---" % input_dim)

model = tf.keras.models.Sequential()
model.add(LSTM(88, input_shape=(1, 88)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(rate=0.20))
model.add(tf.keras.layers.RepeatVector(n=1))
model.add(tf.keras.layers.Dense(60, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(LSTM(50, return_sequences=True))
# model.add(tf.keras.layers.Dense(88, activation=tf.nn.softmax))
model.add(tf.keras.layers.Dropout(rate=0.20))
model.add(tf.keras.layers.TimeDistributed(Dense(88, activation=tf.nn.softmax)))

# Compile the model by defining the loss function, optimizer and metrics.
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
auc = tf.keras.metrics.AUC(multi_label=False)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', auc])

# Fit the model to be trained for our preprocessed X_train and y_train samples, for 100 epochs and 200 batch_size and
# verbose=1.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# model.fit(X_train, X_train, epochs=40, batch_size=32, verbose=1, callbacks=[callback])
model.fit(X_train, X_train, epochs=40, batch_size=50, verbose=1, callbacks=[callback], validation_data=(X_test, X_test))

y_pred = model.predict(X_test)
print(y_pred)

# Get the mean absolute error between actual and reconstruction/prediction
prediction_loss = tf.keras.losses.mae(y_pred, X_test)

# Check the prediction loss threshold for 2% of outliers
loss_threshold = np.percentile(prediction_loss, 93)
print(f'The prediction loss threshold for 10% of outliers is {loss_threshold:.2f}')

# Visualize the threshold
sns.histplot(prediction_loss, bins=30, alpha=0.8)
plt.axvline(x=loss_threshold, color='orange')
plt.show()

# Check the model performance at 2% threshold
threshold_prediction = [0 if i < loss_threshold else 1 for i in prediction_loss]
# # Check the prediction performance
print(classification_report(y_test, threshold_prediction, digits=4))

# confusion_matrix generation and print to the terminal console.
conf_matrix = confusion_matrix(y_test, threshold_prediction)
print(confusion_matrix(y_test, threshold_prediction))

cm = conf_matrix
sns.set(font_scale=1.75)
# x_axis_labels = ["Positive", "Negative"] # labels for x-axis
# y_axis_labels = ["Positive", "Negative"] # labels for y-axis
# x_axis_labels = ["0", "1", "2"]  # labels for x-axis
# y_axis_labels = ["0", "1", "2"]  # labels for y-axis
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

# The AUC score is calculated and printed.
# function for scoring roc auc score for multi-class
lb = LabelBinarizer()
lb.fit(y_test)
y_test = lb.transform(y_test)
y_pred = lb.transform(threshold_prediction)
print("ROC AUC score:", roc_auc_score(y_test, y_pred, average="macro"))

RocCurveDisplay.from_predictions(y_test, y_pred)
plt.show()

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the experiment was: ")
print("--- %s seconds ---" % timeDuration)