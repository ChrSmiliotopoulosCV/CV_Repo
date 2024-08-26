# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
import keras.backend as K

# Train-Test
from sklearn.model_selection import train_test_split

# Scaling data
from sklearn.preprocessing import StandardScaler

# Classification Report
from sklearn.metrics import classification_report, accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

# Remove Missing Values
na = pd.notnull(df["Position"])
df = df[na]

df = df[["Position", 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
         'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
         'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
         'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
         'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
         'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
         'GKKicking', 'GKPositioning', 'GKReflexes']]

print(df)

forward_player = ["ST", "LW", "RW", "LF", "RF", "RS", "LS", "CF"]
midfielder_player = ["CM", "RCM", "LCM", "CDM", "RDM", "LDM", "CAM", "LAM", "RAM", "RM", "LM"]
defender_player = ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"]

df.loc[df["Position"] == "GK", "Position"] = 0

df.loc[df["Position"].isin(defender_player), "Position"] = 1

df.loc[df["Position"].isin(midfielder_player), "Position"] = 2

df.loc[df["Position"].isin(forward_player), "Position"] = 3

x = df.drop("Position", axis=1)

sc = StandardScaler()
x = pd.DataFrame(sc.fit_transform(x))
y = df["Position"]

y_cat = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x.values, y_cat, test_size=0.2)

model = Sequential()
model.add(Dense(60, input_shape=(33,), activation="relu"))
model.add(Dense(15, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="softmax"))
model.compile(Adam(lr=0.01), "categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit(x_train, y_train, verbose=1, epochs=10, batch_size=64)

# y_pred = model.predict(x_test)
# # y_test_class = np.argmax(y_test, axis=1)
# confusion_matrix(y_test, y_pred)

# print(classification_report(y_test_class, y_pred_class))

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

y_pred = model.predict(x_test)
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

print(confusion_matrix(test, pred))
print(classification_report(test, pred))

# print() of the classification_report() with two different formats.
print(classification_report(test, pred, output_dict=True, digits=4))
print(classification_report(test, pred, labels=[0, 1, 2], digits=4))


