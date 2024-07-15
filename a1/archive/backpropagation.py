import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.array([[0.2, -0.3], [0.4, 0.1]])
        self.weights_hidden_output = np.array([[0.7, 0.5], [-0.6, 0.2]])

        # Initialize the biases
        # self.bias_hidden = np.zeros((1, self.hidden_size))
        # self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        x = self.sigmoid(x)
        return x * (1 - x)

    def feedforward(self, X):
        # Input to hidden
        self.hidden_activation = np.dot(
            X, self.weights_input_hidden)
        self.hidden_output = self.sigmoid(self.hidden_activation) + X
        print('z1', self.hidden_output)

        # Hidden to output
        self.output_activation = np.dot(
            self.hidden_output, self.weights_hidden_output)
        self.predicted_output = self.sigmoid(
            self.output_activation) + self.hidden_output
        print('z2', self.predicted_output)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Compute the output layer error
        output_error = 2 * (self.predicted_output - y)
        output_delta = output_error * \
            self.sigmoid_derivative(self.output_activation)
        print('d2', output_delta)

        # Compute the hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * \
            self.sigmoid_derivative(self.hidden_activation)
        print('d1', hidden_delta)

        # Update weights and biases
        self.weights_hidden_output -= np.dot(
            self.hidden_output.T, output_delta) * learning_rate
        print('W2', self.weights_hidden_output)
        # self.bias_output += np.sum(output_delta,
        #                            axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden -= np.dot(X.T, hidden_delta) * learning_rate
        print('W1', self.weights_input_hidden)
        # self.bias_hidden += np.sum(hidden_delta,
        #                            axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            if epoch % 4000 == 0:
                loss = np.sum(np.square(y - output))
                print(f"Epoch {epoch}, Loss:{loss}")


X = np.array([[0.5, 1]])
y = np.array([[1, 0]])

nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
nn.train(X, y, epochs=1, learning_rate=0.1)

# Test the trained model
# output = nn.feedforward(X)
# print("Predictions after training:")
# print(output)
