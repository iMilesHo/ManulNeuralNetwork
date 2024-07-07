# This is a sample Python script.
import math
import numpy as np

# my manul neural network
# •	One input layer with n_0 inputs.
# •	k hidden layer with n_i neurons (i = 1, 2, …, k).
# •	n_(k+1) output layer with n_(k+1) outputs.
# •	Each neuron in the hidden layer and output layer has a bias.
# •	Each neuron in the hidden layer and output layer uses the sigmoid activation function.
# •	The weights are initialized randomly.
# The notation is as follows:
# •	x is the input vector.
# •	t is the target vector.
# •	W_i is the weight matrix of the i-th layer.
# •	b_i is the bias vector of the i-th layer.
# •	z_i is the output vector of the i-th layer before activation function.
# •	a_i is the output vector of the i-th layer after activation function.
# •	o is the output vector of the network.
# •	σ is the sigmoid function.
# •	η is the learning rate.
# •	L is the loss function.
# •	∇ is the gradient.
# •	⊙ is the element-wise multiplication.
# •	⊗ is the dot product.
# •	∂ is the partial derivative.
# The forward propagation calculation is as follows:
# •	z_1 = W_1 ⊗ x + b_1
# •	a_1 = σ(z_1)
# •	z_2 = W_2 ⊗ a_1 + b_2
# •	a_2 = σ(z_2)
# •	...
# •	z_k = W_k ⊗ a_(k-1) + b_k
# •	a_k = σ(z_k)
# •	o = a_k
# The loss function is as follows:
# •	L = 1/m * sum((o - t)^2)
class OneHiddenLayerNeuralNetwork:
    def __init__(self, input, hiddenNeurons, target, learningRate=0.1):
        self.Z1 = None
        # judge if the learningRate is a float
        if type(learningRate) != float:
            raise TypeError('The type of learningRate should be float');
        if type(input) != np.ndarray:
            raise TypeError('The type of input should be numpy.ndarray');
        # judge if the input is a vector
        if len(input.shape) != 1:
            raise TypeError('The input should be a vector');
        # judge if the type is numpy.ndarray
        if type(target) != np.ndarray:
            raise TypeError('The type of target should be numpy.ndarray');
        # judge if the target is a vector
        if len(target.shape) != 1:
            raise TypeError('The target should be a vector');
        # judge if the hiddenNeurons is a integer
        if type(hiddenNeurons) != int:
            raise TypeError('The type of hiddenNeurons should be int');

        self.lr = learningRate

        # the length of the input
        self.n_0 = len(input)

        # concatenate the bias
        self.input_with_bias = np.insert(input, 0, 1)

        # the target is a scalar
        self.target = target

        # the weights matrix of the first hidden layer. The shape is hiddenNeurons * (input + 1)
        self.W1_with_bias = np.random.rand(hiddenNeurons, (self.n_0 + 1))

        # the weights matrix of the second hidden layer. The shape is len(target) * (hiddenNeurons + 1)
        self.W2_with_bias = np.random.rand(len(target), hiddenNeurons + 1)

        # the final output
        self.output = None

    # the activation function is sigmoid function
    def sigmoid(self, x):
        # calculate the by the np array
        return 1 / (1 + np.exp(-x))

    def forward(self):
        # the output of the first hidden layer
        self.z1 = np.dot(self.W1_with_bias, self.input_with_bias)

        # the output of the first hidden layer after activation function
        self.a1 = self.sigmoid(self.z1)

        # concatenate the bias
        self.a1_with_bias = np.insert(self.a1, 0, 1)

        # the output of the second hidden layer
        self.z2 = np.dot(self.W2_with_bias, self.a1_with_bias)

        # the output of the second hidden layer after activation function
        self.output = self.sigmoid(self.z2)

    # loss function is MSE (Mean Squared Error), which is 1/m * sum((output - target)^2)
    # if the number of output is m, the loss function is 1/m * sum((output - target)^2)
    # the derivative of the loss function is 1/m * sum(2 * (output - target)), which is 2/m * sum(output - target)
    def loss(self):
        return np.sum(np.square(self.output - self.target))

    # the backword propagation calculation
    # The derivative of the sigmoid function is f'(x) = f(x) * (1 - f(x))
    # The derivative of the loss function is 2/m * sum(output - target)
    def backword(self):
        # Number of outputs, for scaling the loss derivative
        m = self.target.size

        # ∂l/∂w2 = ∂l/∂output * ∂output/∂z2 * ∂z2/∂w2
        # ∂l/∂b2 = ∂l/∂output * ∂output/∂z2 * ∂z2/∂b2

        # ∂l/∂output = 2/m * sum(output - target), the shape is (len(target), 1)
        dl_doutput = 2 * (self.output - self.target) / m

        # ∂output/∂z2 = output * (1 - output), the shape is (len(target), 1)
        doutput_dz2 = self.output * (1 - self.output)

        # ∂z2/∂w2 = a1, the shape is (hiddenNeurons, 1)
        dz2_dw2 = self.a1_with_bias[1:]

        # b2 is the bias of the second hidden layer, the shape is (len(target), 1), it is located at the first column
        # of W2_with_bias ∂z2/∂b2 = np.ones((hiddenNeurons, 1)) the shape is (hiddenNeurons, 1), which every element
        # is 1
        dz2_db2 = np.ones(len(self.target))

        # so dl_dw2 is as follows, the shape is (len(target), hiddenNeurons) which is the same as W2
        dl_dw2 = np.outer(dl_doutput * doutput_dz2, dz2_dw2)

        # so dl_db2 is as follows, the shape is (len(target), 1) which is the same as b2
        dl_db2 = dl_doutput * doutput_dz2 * dz2_db2

        #update the weights of the second hidden layer, which is W2_with_bias
        # W2_with_bias = W2_with_bias - learningRate * (dl_dw2 concatenate dl_db2)
        # concatenate dl_db2 to dl_dw2, the shape is (len(target), hiddenNeurons + 1) in which the first column is dl_db2
        update_W2_with_bias = np.hstack((dl_db2.reshape(-1, 1), dl_dw2))
        self.W2_with_bias = self.W2_with_bias - self.lr * update_W2_with_bias

        # ∂l/∂w1 = ∂l/∂output * ∂output/∂z2 * ∂z2/∂a1 * ∂a1/∂z1 * ∂z1/∂w1
        # ∂l/∂b1 = ∂l/∂output * ∂output/∂z2 * ∂z2/∂a1 * ∂a1/∂z1 * ∂z1/∂b1

        # ∂l/∂output * ∂output/∂z2 is the same as the calculation of ∂l/∂w2
        # ∂z2/∂a1 = W2, the shape is (len(target), hiddenNeurons)
        dz2_da1 = self.W2_with_bias[:, 1:]

        # ∂a1/∂z1 = a1 * (1 - a1), the shape is (hiddenNeurons, 1)
        da1_dz1 = self.a1 * (1 - self.a1)

        # ∂z1/∂w1 = x, the shape is (n_0, 1)
        dz1_dw1 = self.input_with_bias[1:]

        # b1 is the bias of the first hidden layer, the shape is (hiddenNeurons, 1),
        # it is located at the first column of W1_with_bias
        # ∂z1/∂b1 = np.ones((hiddenNeurons, 1)) the shape is (hiddenNeurons, 1), which every element is 1
        dz1_db1 = np.ones(self.W1_with_bias.shape[0])

        # so dl_dw1 is as follows, the shape is (hiddenNeurons, n_0) which is the same as W1
        dl_dw1 = np.outer(np.dot(dl_doutput * doutput_dz2, dz2_da1)*da1_dz1, dz1_dw1)

        # so dl_db1 is as follows, the shape is (hiddenNeurons, 1) which is the same as b1
        dl_db1 = np.dot(dl_doutput * doutput_dz2, dz2_da1)*da1_dz1 * dz1_db1

        #update the weights of the first hidden layer, which is W1_with_bias
        # W1_with_bias = W1_with_bias - learningRate * (dl_dw1 concatenate dl_db1)
        # concatenate dl_db1 to dl_dw1, the shape is (hiddenNeurons, n_0 + 1) in which the first column is dl_db1
        update_W1_with_bias = np.hstack((dl_db1.reshape(-1, 1), dl_dw1))
        self.W1_with_bias = self.W1_with_bias - self.lr * update_W1_with_bias

    def train(self, epochs):
        for i in range(epochs):
            self.forward()
            self.backword()
            print('epoch: ', i, ' loss: ', self.loss())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # the input vector
    input = np.array([1, 2, 3])
    # the target vector
    target = np.array([1, 0])
    # the number of hidden neurons
    hiddenNeurons = 4
    # the learning rate
    learningRate = 0.1
    # the number of epochs
    epochs = 100000
    # create the neural network
    nn = OneHiddenLayerNeuralNetwork(input, hiddenNeurons, target, learningRate)
    # train the neural network
    nn.train(epochs)
    # print the output
    print('output: ', nn.output)
    # print the loss
    print('loss: ', nn.loss())

