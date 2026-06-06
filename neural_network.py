import numpy as np
import matplotlib.pyplot as plt

xTrain = np.loadtxt('Data/X_train.csv')
yTrain = np.loadtxt('Data/Y_train.csv')
xTest = np.loadtxt('Data/X_test.csv')
yTest = np.loadtxt('Data/Y_test.csv')

yTrain = yTrain.reshape(-1, 1)
yTest = yTest.reshape(-1, 1)


xMean = xTrain.mean(axis=0)
xStd = xTrain.std(axis=0)
xTrainNorm = (xTrain - xMean) / xStd
xTestNorm = (xTest - xMean) / xStd

yMean = yTrain.mean()
yStd = yTrain.std()
yStd = 1 if yStd == 0 else yStd
yTrainNorm = (yTrain - yMean) / yStd
yTestNorm = (yTest - yMean) / yStd

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, lr = 0.01, activation = "sigmoid"):
        self.lr = lr
        self.activation = activation
        
        # 1 is input and 2 is output
        self.W1 = np.random.rand(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
        self.lossHistory = []
        self.accHistory = []

    def sigmoid(self, x, derivative=False):
        if derivative:
            s = 1/(1 + np.exp(-x))
            return s * (1 - s)
        return 1/(1 + np.exp(-x))
    
    def relu(self, x, derivative=False):
        if derivative:
            return (x>0).astype(float)
        return np.maximum(0, x)
    
    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - np.tanh(x) ** 2
        return np.tanh(x)

    def linear(self, x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x

    def activate(self, x, derivative=False):
        if self.activation == "sigmoid":
            return self.sigmoid(x, derivative)
        elif self.activation == 'tanh':
            return self.tanh(x, derivative)
        elif self.activation == 'relu':
            return self.relu(x, derivative)
        elif self.activation == 'linear':
            return self.linear(x, derivative)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, y):
        n = X.shape[0]

        dz2 = 2 * (self.z2 - y) / n
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activate(self.z1, derivative=True)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            yPred = self.forward(X)

            loss = np.mean((yPred - y) ** 2)
            self.lossHistory.append(loss)

            ssRes = np.sum((y - yPred) ** 2)
            ssTot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ssRes / (ssTot + 1e-10))
            self.accHistory.append(r2)

            self.backward(X, y)

    def evaluate(self, X, y):
        yPred = self.forward(X)
        loss = np.mean((yPred - y) ** 2)
        ssRes = np.sum((y - yPred) ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ssRes / (ssTot + 1e-10))
        return loss, r2
    

input_size = xTrainNorm.shape[1]

#when the number of neurons is set to 1 I got a total loss of .9172 and r2 of .0574, when I set it to 10 neuron 
#I got a loss of .7855 and r2 of .1928. This would prove that the model when given more neurons in the hidden layer
#is more accurate however both models are still underfitting the data as the loss is still high and r2 is low.
#This was tested with a consisten lr or learning rate of .01 1000 epochs and sigmoid activation function.
hidden_size = 10
nn = NeuralNetwork(input_size, hidden_size, lr=0.01, activation="linear")
#(hidden size set to 10 meaining tests were run with 10 neurons)when the activation is set to sigmoid we get a Test Loss: 0.7397 and Test R2: 0.2399, which does not show great performance.
#When using tanh we get a Test Loss: 0.0038 and Test R2: 0.9961 this was the best non-linear activation, 
#When using relu we get a Test Loss: 0.0656 and Test R2: 0.9326, this activation performs well but is slightly worsre than the tanh activation.
#When we use a linear activation we get Test Loss: 0.0000 and Test R2: 1.0000 which means the data is either perfectly linear or the model is overfitting the data.

nn.train(xTrainNorm, yTrainNorm, epochs=1000)
testloss, testR2 = nn.evaluate(xTestNorm, yTestNorm)
print(f"Activation type:{nn.activation}")
print(f"Neurons:{hidden_size}")
print(f"Test Loss: {testloss:.4f}")
print(f"Test R2: {testR2:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(nn.lossHistory)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(nn.accHistory)
plt.xlabel('Epochs')
plt.ylabel('R2 Score')
plt.title('Training Accuracy (R2 Score)')
plt.grid(True)

plt.tight_layout()
plt.show()
