# Import Classes and Functions
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# Encode the Output Variable
# When modeling multi-class classification problems using neural networks, it is good practice to reshape the output
# attribute from a vector that contains values for each class value to a matrix with a Boolean for each class value
# and whether a given instance has that class value or not.
# This is called one-hot encoding or creating dummy variables from a categorical variable.

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# You can now create your KerasClassifier for use in scikit-learn.

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)

# Evaluate the Model with k-Fold Cross Validation
# You can now evaluate the neural network model on our training data. The scikit-learn has excellent capability to
# evaluate models using a suite of techniques. The gold standard for evaluating machine learning models is k-fold
# cross validation.

kfold = KFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

