# ManulNeuralNetwork

A Manul Neural Network Implementation

• One input layer with n*0 inputs.
• k hidden layer with n_i neurons (i = 1, 2, …, k).
• n*(k+1) output layer with n\_(k+1) outputs.
• Each neuron in the hidden layer and output layer has a bias.
• Each neuron in the hidden layer and output layer uses the sigmoid activation function.
• The weights are initialized randomly.

## The notation is as follows:

• x is the input vector.
• t is the target vector.
• W_i is the weight matrix of the i-th layer.
• b_i is the bias vector of the i-th layer.
• z_i is the output vector of the i-th layer before activation function.
• a_i is the output vector of the i-th layer after activation function.
• o is the output vector of the network.
• σ is the sigmoid function.
• η is the learning rate.
• L is the loss function.
• ∇ is the gradient.
• ⊙ is the element-wise multiplication.
• ⊗ is the dot product.
• ∂ is the partial derivative.

## The forward propagation calculation is as follows:

• z*1 = W_1 ⊗ x + b_1
• a_1 = σ(z_1)
• z_2 = W_2 ⊗ a_1 + b_2
• a_2 = σ(z_2)
• ...
• z_k = W_k ⊗ a*(k-1) + b_k
• a_k = σ(z_k)
• o = a_k
The loss function is as follows:
• L = 1/m \* sum((o - t)^2)
