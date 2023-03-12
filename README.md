<div align="center">
<img src="./images/CL Banner.png"/>
</div>

# Basic Multi-layer Perceptron Neural Network

### A simple multi-layer perceptron neural network built from scratch

This code implements a basic multi-layer perceptron (MLP) neural network from scratch. The purpose of this is to illustrate the inner workings of a neural network by building one from the ground up and then demonstrate how easily it can be replicated when utilizing TensorFlow.

**NOTE: This code is a modified version of code from [Valerio Velardo's (The Sound of AI)](https://www.youtube.com/@ValerioVelardoTheSoundofAI) [source code](https://github.com/musikalkemist/DeepLearningForAudioWithPython). Please see license for copyright information.**

## Motivation
When working with machine learning (ML) models, we use pre-built libraries with immense functionality all of the time. While it is efficient to utilize what has already been constructed (why reinvent the wheel?), this never truly imparts an understanding of the math that lives under the hood. Without an understanding of the components of a system, you can never utilize it to its full potential. The motivation for replicating this code was to provide a deeper understanding for myself and others of a simple neural net so that each element is more tangible.

## MLP Neural Network Background

An MLP is a type of artificial neural network that consists of multiple layers of interconnected nodes, also known as neurons. It is a feedforward neural network, which means that the input data flows through the network in one direction, and the output is generated at the end of the network.

<div align="center">
<img src="./images/MLP.jpg" width=500/>

<em>(source: https://www.geeksforgeeks.org/multi-layer-perceptron-learning-in-tensorflow/)</em>
</div>

The architecture of an MLP includes an input layer, hidden layers, an output layer, activation functions, and weights and biases.

- **Input Layer:** The input layer is the first layer of the MLP, where the raw input data is fed into the network. Each neuron in the input layer corresponds to one input feature of the data. The neurons in this layer do not perform any computation and simply pass the input data to the next layer.

- **Hidden Layers:**  The hidden layers are the layers between the input and output layers, where the computation of the network occurs. Each neuron in the hidden layers receives input from the neurons in the previous layer, computes a weighted sum of these inputs, applies an activation function, and outputs the result to the next layer. The number of hidden layers and the number of neurons in each hidden layer are hyperparameters that can be tuned to improve the performance of the MLP.

- **Output Layer:** The output layer is the last layer of the MLP, where the final output of the network is generated. The number of neurons in the output layer depends on the type of problem the MLP is being used to solve. For example, in a speech recognition task, the output layer may have neurons corresponding to phonemes or words. In a music genre classification task, the output layer may have neurons corresponding to different genres.

- **Activation Functions:** Activation functions are used in the hidden layers to introduce nonlinearity into the network, which allows it to model complex relationships between the input and output. A range of activation functions can be utilized, but in our case, we are specifically implementing the sigmoid function.

- **Weights and Biases:** Each neuron in the MLP is associated with a set of weights and biases, which are learned during the training process. The weights determine the strength of the connections between the neurons, while the biases control the threshold at which the neuron activates. The learning algorithm, gradient descent in our model, adjusts the weights and biases to minimize the error between the predicted output and the actual output, which allows the MLP to learn to make accurate predictions on new data. Our model implements a backpropagation optimization algorithm to calculate errors to influence the gradient descent. A common error metric, such as mean squared error (MSE), can be used to measure the performance of the model.

MLPs are used in a variety of applications, including image and speech recognition, natural language processing (NLP), and predictive modeling. They are known for their ability to handle complex nonlinear relationships in data and can be used for both regression and classification tasks. However, they can be computationally expensive to train and require large amounts of data to achieve good performance.

## Structure
The MLP (Multilayer Perceptron Class) is broken up into a constructor and 7 separate methods:
- **Constructor** - Constructor for the MLP. Takes the number of inputs, a variable number of hidden layers, and number of outputs.
  - *num_inputs (int)* - Number of inputs
  - *hidden_layers (list)* - A list of ints for the hidden layers
  - *num_outputs (int)* - Number of outputs
- **forward_propagate** - Computes the forward propagation  of the network based on input signals.
  - *Parameters:*
    - *inputs (ndarray)* - Input signals
  - *Returns:*
    - *activations (ndarray)* - Output values
- **back_propagate** - Backpropogates an error signal.
    - *Parameters:*
        - *error (ndarray)* - The error to backpropagate
    - *Returns:*
        - *error (ndarray)* - The final error of the input
- **train** - Trains model through forward propagation and backpropagation with gradient descent of mse
    - *Parameters:*
        - *inputs (ndarray)* - X (features)
        - *targets (ndarray)* - Y (target)
        - *epochs (int)* - Number of epochs the network is to be trained for
        - *learning_rate (float)* - The rate (or step) to apply to gradient descent
- **gradient_descent** - Learns (updates weights) by descending the error gradient
    - *Parameters:*
        - *learningRate (float)* - The rate (or step) at which the model will learn
- **_sigmoid** - Sigmoid activation function $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
    - *Parameters:*
        - *x (float)* - Value to be processed through sigmoid function
    - *Returns:*
        - *y (float)* - Calculated sigmoid function result
- **_sigmoid_derivative** - Sigmoid derivative function $$\sigma'(x) = \sigma(x)*(1 - \sigma(x))$$
    - *Arguments:*
        - *x (float)* - Value to be processed through sigmoid derivative
    - *Returns:*
        - *y (float)* - Calculated sigmoid derivative result
- **_mse** - Mean Squared Error loss function $$MSE = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$$
    - *Parameters:*
        - *target (ndarray)* - The ground truth
        - *output (ndarray)* - The predicted values
    - *Returns:*
        - *(float)* - Calculated mean squared error


## More Information 
If you like what is here and are interested in audio machine learning, follow Valerio on his incredible YouTube channel at [The Sound of AI](https://www.youtube.com/@ValerioVelardoTheSoundofAI).

I am an audio researcher and scientist with a passion for music, spatial audio, and machine learning. You can see more of my work on [Google Scholar](https://scholar.google.com/citations?hl=en&view_op=list_works&gmla=AJsN-F6PaFcTdi4cTxZ3Kpvf2xwKM4ramDbqVKFm_buMLElpYMNzxViHQuKgOPeLMMP3KkcK6besvk4Tu9wURTx-4smBAfXZtw&user=4K5CzM4AAAAJ) and my [GitHub page](https://github.com/crlandsc).

Current Research: [Binaural Externalization Processing](https://www.chrislandschoot.com/binaural-externalization)

I also make music under the name [After August](https://www.after-august.com/). Check me out on [Spotify](https://open.spotify.com/artist/2i6noWJnJQPXPsudoiJuMS?si=AOMNQvWgQESKoKooa9qeAw) and [YouTube](https://youtube.com/@AfterAugust)!