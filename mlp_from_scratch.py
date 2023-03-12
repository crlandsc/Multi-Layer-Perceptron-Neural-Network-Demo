import numpy as np
from random import random

"""This is sample code to demonstrate the inner-workings of a neural network.
Below we build a Multilayer Perceptron (MLP) Feedforward Neural Network.

We then test the neural net with two input numbers when added together equal the output number.
This trains the model to learn that adding these two numbers together will equal the output number.
The error values are printed for each learning step to see how the model is learning.

Code modified from Sound of AI source: https://github.com/musikalkemist/DeepLearningForAudioWithPython
See LICENSE for details.
"""

class MLP:
    """A Multilayer Perceptron neural network class.
    """

    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Parameters:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        layers = [num_inputs] + hidden_layers + [num_outputs]

        # initiate random weights
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i+1])
            weights.append(w)  # append weight matrix
        self.weights = weights

        # Derivatives
        derivatives = []
        for i in range(len(layers) - 1):  # same number as weight matrices (in between layers)
            d = np.zeros((layers[i], layers[i+1]))  # create empty matrix of derivatives between neurons
            derivatives.append(d)
        self.derivatives = derivatives

        # Activations
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])  # create empty array of activations for neurons
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Parameters:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """
        activations = inputs  # the input layer activation = itself
        self.activations[0] = inputs  # activations for the first layer, backpropagation

        # iterate through the network layers
        for i, w in enumerate(self.weights):
            # calculate net inputs between previous activation and weight matrix
            net_inputs = np.dot(activations, w)  # .dot() performs matrix multiplication

            # calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations  # save to the next layer, backpropagation (see example equations below)

        return activations  # return output layer activation

    def back_propagate(self, error, verbose=False):
        """Backpropagates an error through the network
        Parameters:
            error (ndarray): The error to backpropagate
        Returns:
            error (ndarray): The final error of the input
        """

        for i in reversed(range(len(self.derivatives))):
            # Get activation for previous layer
            activations = self.activations[i+1]

            # Apply sigmoid derivative function
            delta = error * self._sigmoin_derivative(activations)

            # Reshape delta to a 2D array
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            # Get activations for current layer
            current_activations = self.activations[i]

            # Reshape activations to a 2Dd column matrix
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            # Save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)

            # Backpropogate the next error
            error = np.dot(delta, self.weights[i].T)

            # Print derivatives for testing
            if verbose:
                print(f"Derivatives for W{i}: {self.derivatives[i]}")

        return error

    def train(self, inputs, targets, epochs, learning_rate):
        """Trains model through forward propagation and backpropagation with gradient descent of mse
        Parameters:
            inputs (ndarray): X (features)
            targets (ndarray): y (target)
            epochs (int): Number of epochs the network is to be trained for
            learning_rate (float): The rate (or step) to apply to gradient descent
        """
        # Outer loop - train for the number of epochs specified
        for i in range(epochs):
            sum_errors = 0  # reset errors to 0 for each epoch

            # Inner loop - iterate through the training data
            for j, input in enumerate(inputs):
                target = targets[j]

                # Activate network (send inputs through network)
                output = self.forward_propagate(input)

                # Calculate error
                error = target - output

                # Backpropagate (calculate error gradients with respect to the weights)
                self.back_propagate(error, verbose=False)

                # Gradient descent (update weights)
                self.gradient_descent(learning_rate)

                # keep track of the MSE for reporting later
                sum_errors += self._mse(target, output)

            # Epoch complete, report the training error
            print("Error: {} at epoch {}".format(sum_errors / len(items), i+1))

        print("Training complete!")
        print("=====")

    def gradient_descent(self, learning_rate=1):
        """Learns (updates weights) by descending the error gradient
        Parameters:
            learning_rate (float): The rate (or step) at which the model will learn
        """
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate

    def _sigmoid(self, x):
        """Sigmoid derivative function
        Parameters:
            x (float): Value to be processed through sigmoid function
        Returns:
            y (float): Calculated sigmoid function result
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoin_derivative(self, x):
        """Sigmoid derivative function
        Parameters:
            x (float): Value to be processed through sigmoid derivative
        Returns:
            y (float): Calculated sigmoid derivative result
        """
        return x * (1.0 - x)

    def _mse(self, target, output):
        """Mean Squared Error loss function
        Parameters:
            target (ndarray): The ground truth
            output (ndarray): The predicted values
        Returns:
            (float): Calculated mean squared error
        """
        return np.average((target - output) ** 2)


if __name__ == "__main__":
    # Create training dataset with 5000 samples
    items = np.array([[random()/2 for _ in range(2)] for _ in range(5000)])
    targets = np.array([[i[0] + i[1]] for i in items])

    # Create a Multilayer Perceptron with one hidden layer
    #   input layer: 2 neurons
    #   hidden layer: 5 neurons (can add to list create arbitrary number of hidden layers
    #   output layer: 1 neuron
    mlp = MLP(2, [5], 1)

    # Train network
    mlp.train(items, targets, 100, 0.1)  # 100 epochs with a learning rate of 0.1

    # Create test data
    input = np.array([0.1, 0.3])
    target = np.array([0.4])

    # Make predictions
    output = mlp.forward_propagate(input)
    print(f"Trained Neural Network Calculation Test: {input[0]} + {input[1]} is equal to {output[0]}")