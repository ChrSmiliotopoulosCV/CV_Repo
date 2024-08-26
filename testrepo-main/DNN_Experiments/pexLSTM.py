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
path = r"D:\Personal Projects\Διδακτορικό\Paper_no2\Current Work Fld [Feb 2023]\ExtractedFiles [07 March " \
       r"2023]\Experimental Results\DNN Experiments Results\LSTM(5hidden - 20epochs - 32batch).txt"
sys.stdout = Logger(path)

# Print the version of the sklearn library, for reasons of compatibility.
sklearn_version = sklearn.__version__
print(sklearn_version)
print(pd.__version__)

# Importing the Dataset
# Read the original and labeled CSV file into the pandas 'df' dataframe
df = pd.read_csv(
    (
        r"D:\Personal Projects\Διδακτορικό\Paper_no2\Current Work Fld [Feb 2023]\ExtractedFiles [07 March "
        r"2023]\newExtendedSchema [From Jan 23]\Preprocessing_Dataframes [Final "
        r"Extraction]\1.75M_FinallySelected\full-csv(Evaluated Titles).csv"
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

# Keras requires your output feature to be one-hot encoded values.
Y_final = tf.keras.utils.to_categorical(y)
print("Therefore, our final shape of output feature will be {}".format(Y_final.shape))

# Divide dataframe into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, Y_final, random_state=42)
print("The length of the X_train set is: ", len(X_train))
print("The length of the X_test set is: ", len(X_test))
print("The length of the y_train set is: ", len(y_train))
print("The length of the y_test set is: ", len(y_test))

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
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

# Compile the model by defining the loss function, optimizer and metrics.
# model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
auc = tf.keras.metrics.AUC(multi_label=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy', auc])

# Fit the model to be trained for our preprocessed X_train and y_train samples, for 100 epochs and 200 batch_size and
# verbose=1.
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, callbacks=[callback])

# Check the summary of the train and fitted model
model.summary()
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

y_pred = model.predict(X_test)
# Converting predictions to label
pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
# Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

a = accuracy_score(pred, test)
print('Accuracy is:', a * 100)

# confusion_matrix generation and print to the terminal console.
conf_matrix = confusion_matrix(test, pred)
print(confusion_matrix(test, pred))

# print() of the classification_report() with two different formats.
print(classification_report(test, pred))
print(classification_report(test, pred, output_dict=True, digits=4))
print(classification_report(test, pred, labels=[0, 1, 2], digits=4))

# The duration of the experiments time is calculated and printed on the terminal's screen.
timeDuration = time.time() - start_time
print("The time duration of the experiment was: ")
print("--- %s seconds ---" % timeDuration)

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
