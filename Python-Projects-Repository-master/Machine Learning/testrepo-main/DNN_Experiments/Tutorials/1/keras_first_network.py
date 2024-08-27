# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:, 0:8]
y = dataset[:, 8]

# Define Keras Model
# We create a Sequential model and add layers one at a time until we are happy with our network architecture.
# The first thing to get right is to ensure the input layer has the correct number of input features. This can be
# specified when creating the first layer with the input_shape argument and setting it to (8,) for presenting the
# eight input variables as a vector.

# In this example, let’s use a fully-connected network structure with three layers.
# Fully connected layers are defined using the Dense class. You can specify the number of neurons or nodes in the
# layer as the first argument and the activation function using the activation argument.

# It used to be the case that Sigmoid and Tanh activation functions were preferred for all layers. These days, better
# performance is achieved using the ReLU activation function. Using a sigmoid on the output layer ensures your network
# output is between 0 and 1 and is easy to map to either a probability of class 1 or snap to a hard classification of
# either class with a default threshold of 0.5.

# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Note:  The most confusing thing here is that the shape of the input to the model is defined as an argument on the
# first hidden layer. This means that the line of code that adds the first Dense layer is doing two things, defining
# the input or visible layer and the first hidden layer.

# Compile Keras Model
# Compiling the model uses the efficient numerical libraries under the covers (the so-called backend) such as Theano
# or TensorFlow. The backend automatically chooses the best way to represent the network for training and making
# predictions to run on your hardware, such as CPU, GPU, or even distributed.

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))