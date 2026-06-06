# Neural-Network
A 3 layer feedforward neural network using forward propagation, backpropagation and gradient descent.

## How the NN works
Input -> Hidden -> Output make up the 3 layers of the network.
The network learns through
- Forward propagation, matrix multiplications with activation functions
- Loss calculation, finding the mean squared error between predictions and actual values
- Backpropagation, computing gradients for each weight
- Gradient descent, updating weights and biases to minimize loss
- Evaluation, tracking both loss and R^2 score over time

## Features
- Uses 4 different activation functions ( sigmoid, relu, tanh, and linear) each having manual derivative implementations
- Manual z-score normalization of input and output data
- Tracks training loss and r^2 across epochs
- Visualizes performance with Matplotlib
- Separate forward(), backward(), train(), and evaluate() methods

## Results
Found within code comments

## Stack
Python, NumPy, Matplotlib
