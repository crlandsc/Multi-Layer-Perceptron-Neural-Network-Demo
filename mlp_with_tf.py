# Imports
import numpy as np
from random import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# import tensorflow as tf
from sklearn.model_selection import train_test_split

"""This is sample code to demonstrate how to construct a basic neural network with TensorFlow.
Below we build a Multilayer Perceptron (MLP) Dense (aka Feedforward) Neural Network.

We then test the neural net with two input numbers when added together equal the output number.
This trains the model to learn that adding these two numbers together will equal the output number.
The error values are printed for each learning step to see how the model is learning.

Code modified from Sound of AI source: https://github.com/musikalkemist/DeepLearningForAudioWithPython
See LICENSE for details.
"""


def generate_dataset(num_samples, test_size):
    """Generates train-test data for neural network
    Parameters:
        num_samples (int): Total number of samples in dataset
        test_size (int): Ratio of num_samples used as test set (for train-test split)

    Returns:
        X_train (ndarray): Input data for training (2D array)
        X_test (ndarray): Input data for testing (2D array)
        y_train (ndarray): Target data for training (2D array)
        y_test (ndarray): Target data for testing (2D array)
    """
    # Generate X (inputs) and y (target) data
    # y[0] = X[0] + X[1]
    X = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in X])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Create a dataset of 5000 samples with a train/test split of 70%/30%
    X_train, X_test, y_train, y_test = generate_dataset(5000, 0.3)

    # Instantiate sequential model with tensorflow:
    #   input layer: 2 nodes
    #   hidden layer: 5 nodes
    #   output layer: 1 nodes
    # Note: dense layer connects all nodes (aka neurons) with previous layer
    model = Sequential([
        Dense(5, input_dim=2, activation="relu"),  # Hidden layer (activation = relu, performs better than sigmoid)
        Dense(1, activation="linear")  # Output layer (activation = identity for regression problem)
    ])

    # Compile model
    optimizer = SGD(learning_rate=0.1)  # SDG = Stochastic gradient descent
    model.compile(optimizer=optimizer, loss='mse')  # MSE = Mean-squared-error

    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=100)

    # Evaluate model
    print("\nModel Evaluation:")
    model.evaluate(X_test, y_test, verbose=2)

    # Make predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    print("Predictions:")
    for d, p in zip(data, predictions):
        print(f"{d[0]} + {d[1]} = {p[0]}")