import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target

    enc = OneHotEncoder()
    y = enc.fit_transform(y.reshape(-1, 1)).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(s):
    return s * (1 - s)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_derivative(z):
    z_expanded = z[:, :, np.newaxis]
    identity = np.eye(z.shape[1])
    jacobian_matrix = z_expanded * (identity - z_expanded.transpose(0, 2, 1))
    return jacobian_matrix


# Neural network forward and backward propagation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        self.dW1 = np.zeros((input_size, hidden_size))
        self.db1 = np.zeros((1, hidden_size))
        self.dW2 = np.zeros((hidden_size, output_size))
        self.db2 = np.zeros((1, output_size))

    def forward(self, X):
        # Implement forward pass
        # You should store the intermediate values as fields to be used by the backward pass
        # The forward pass must work for all batch sizes

        # X: the x values, batched along dimension zero
        # return: z2
        self.a1 = np.dot(X, self.W1) + self.b1
        self.z1 = sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = softmax(self.a2)
        return self.z2

    def backward(self, X, y):
        # Implement backward pass, assuming the MSE loss
        # Use the intermediate values calculated from the forward pass,
        # and store the gradients in fields
        # The backward pass only need to work for batch size 1

        # X: the x values, batched along dimension zero
        # y: batched target values
        # return: None
        dz2 = 2 * (self.z2 - y) / y.shape[0]
        d2 = np.einsum('ij,ijk->ik', dz2, softmax_derivative(self.z2))

        # Gradients for W2 and b2
        self.dW2 = np.dot(self.z1.T, d2)
        self.db2 = np.sum(d2, axis=0, keepdims=True)

        d1 = np.dot(d2, self.W2.T) * sigmoid_derivative(self.z1)

        # Gradients for W1 and b1
        self.dW1 = np.dot(X.T, d1)
        self.db1 = np.sum(d1, axis=0, keepdims=True)

    def get_params_and_grads(self):
        # Return parameters and corresponding gradients
        params = [self.W1, self.b1, self.W2, self.b2]
        grads = [self.dW1, self.db1, self.dW2, self.db2]
        return params, grads


def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def compute_accuracy(predictions, targets):
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(targets, axis=1)
    return np.mean(pred_classes == true_classes)


def plot_all_results(all_losses, all_accuracies, all_labels, save_dir='plots', filename='2layers.png'):
    if len(all_losses) != len(all_accuracies):
        raise ValueError(
            "all_losses length must be equal to all_accuracies length")

    if len(all_losses) != len(all_labels):
        raise ValueError(
            "all_labels length must be equal to all_losses length")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = len(all_losses[0])
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for i in range(len(all_losses)):
        plt.plot(range(1, epochs + 1), all_losses[i], label=all_labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss behaviours')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i in range(len(all_losses)):
        plt.plot(range(1, epochs + 1), all_accuracies[i], label=all_labels[i])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy behaviours')
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path)
    plt.close()


# Optimizer implementations (SGD, SGD with Momentum, AdaGrad)
class SGD():
    def __init__(self, params, learning_rate):
        self.params = params
        self.lr = learning_rate

    def step(self, grads):
        # Perform one step of SGD
        for param, grad in zip(self.params, grads):
            param -= self.lr * grad


class SGD_Momentum():
    def __init__(self, params, learning_rate, alpha):
        self.params = params
        self.lr = learning_rate
        self.alpha = alpha
        self.velocity = [np.zeros_like(param) for param in params]

    def step(self, grads):
        # Perform one step of SGD with momentum
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.velocity[i] = self.alpha * self.velocity[i] - self.lr * grad
            param += self.velocity[i]


class SGD_AdaGrad():
    def __init__(self, params, learning_rate, delta):
        self.params = params
        self.lr = learning_rate
        self.delta = delta
        self.r = [np.zeros_like(param) for param in params]

    def step(self, grads):
        # Perform one step of SGD with adagrad
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.r[i] += grad ** 2
            param -= self.lr * grad / (np.sqrt(self.r[i]) + self.delta)


# batch generator
def gen_batches(data, labels, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size], labels[i:i+batch_size]


# Training loop
def train(network, data, optimizer, epochs, batch_size):
    X_train, y_train, X_test, y_test = data
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        random_indices = np.random.permutation(list(range(X_train.shape[0])))
        X_train = X_train[random_indices]
        y_train = y_train[random_indices]
        for x, y in gen_batches(X_train, y_train, batch_size):
            # Forward pass
            output = network.forward(x)

            # Backward pass
            network.backward(x, y)

            # Get parameters and gradients
            params, grads = network.get_params_and_grads()

            # Update parameters using the chosen optimizer
            optimizer.step(grads)

        # Compute loss and accuracy
        X_test = X_train
        y_test = y_train
        output = network.forward(X_test)
        train_loss = mean_squared_error(output, y_test)
        train_accuracy = compute_accuracy(output, y_test)

        test_losses.append(train_loss)
        test_accuracies.append(train_accuracy)

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    return test_losses, test_accuracies


# Main function to run the experiments
def main():
    data = load_data()
    batch_size = 128
    input_size = 64  # For the digits dataset (8x8 images flattened)
    hidden_size = 20
    output_size = 10
    epochs = 200

    print("Training with SGD LR 0.5")
    network = NeuralNetwork(input_size, hidden_size, output_size)
    test_losses_sgd, test_accuracies_sgd = train(network, data, SGD(
        network.get_params_and_grads()[0], 0.5), epochs, batch_size)

    print("Training with SGD LR 0.2")
    network = NeuralNetwork(input_size, hidden_size, output_size)
    test_losses_sgd_b, test_accuracies_sgd_b = train(network, data, SGD(
        network.get_params_and_grads()[0], 0.2), epochs, batch_size)

    print("Training with SGD with Momentum LR 0.05")
    network = NeuralNetwork(input_size, hidden_size, output_size)
    test_losses_momentum, test_accuracies_momentum = train(network, data, SGD_Momentum(
        network.get_params_and_grads()[0], 0.05, 0.9), epochs, batch_size)

    print("Training with AdaGrad LR 0.05")
    network = NeuralNetwork(input_size, hidden_size, output_size)
    test_losses_adagrad, test_accuracies_adagrad = train(network, data, SGD_AdaGrad(
        network.get_params_and_grads()[0], 0.05, 1e-8), epochs, batch_size)

    print("Training with AdaGrad LR 0.01")
    network = NeuralNetwork(input_size, hidden_size, output_size)
    test_losses_adagrad_b, test_accuracies_adagrad_b = train(network, data, SGD_AdaGrad(
        network.get_params_and_grads()[0], 0.01, 1e-8), epochs, batch_size)

    # Compare train losses and train accuracies
    all_losses = [test_losses_sgd, test_losses_sgd_b,
                  test_losses_momentum, test_losses_adagrad, test_losses_adagrad_b]
    all_accuracies = [test_accuracies_sgd, test_accuracies_sgd_b,
                      test_accuracies_momentum, test_accuracies_adagrad, test_accuracies_adagrad_b]
    all_labels = ["SGD LR 0.5", "SGD LR 0.2",
                  "SGD-Momentum LR 0.05", "AdaGrad LR 0.05", "AdaGrad LR 0.01"]
    plot_all_results(all_losses, all_accuracies, all_labels)


if __name__ == "__main__":
    main()


# Please implement the 3-layer neural network and run the main function again to plot the curve.
def relu(z):
    return np.maximum(0, z)


def relu_derivative(s):
    return np.where(s > 0, 1, 0)


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(
            input_size, hidden_size1) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(
            hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(
            hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
        self.b3 = np.zeros((1, output_size))

        self.dW1 = np.zeros((input_size, hidden_size1))
        self.db1 = np.zeros((1, hidden_size1))
        self.dW2 = np.zeros((hidden_size1, hidden_size2))
        self.db2 = np.zeros((1, hidden_size2))
        self.dW3 = np.zeros((hidden_size2, output_size))
        self.db3 = np.zeros((1, output_size))

    def forward(self, X):
        # Implement forward pass
        self.a1 = np.dot(X, self.W1) + self.b1
        self.z1 = sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = sigmoid(self.a2)
        self.a3 = np.dot(self.z2, self.W3) + self.b3
        self.z3 = softmax(self.a3)
        return self.z3

    def backward(self, X, y):
        dz3 = 2 * (self.z3 - y) / y.shape[0]
        d3 = np.einsum('ij,ijk->ik', dz3, softmax_derivative(self.z3))

        # Gradients for W3 and b3
        self.dW3 = np.dot(self.z2.T, d3)
        self.db3 = np.sum(d3, axis=0, keepdims=True)

        d2 = np.dot(d3, self.W3.T) * sigmoid_derivative(self.z2)

        # Gradients for W2 and b2
        self.dW2 = np.dot(self.z1.T, d2)
        self.db2 = np.sum(d2, axis=0, keepdims=True)

        d1 = np.dot(d2, self.W2.T) * sigmoid_derivative(self.z1)

        # Gradients for W1 and b1
        self.dW1 = np.dot(X.T, d1)
        self.db1 = np.sum(d1, axis=0, keepdims=True)

    def get_params_and_grads(self):
        # Return parameters and corresponding gradients
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        grads = [self.dW1, self.db1, self.dW2, self.db2, self.dW3, self.db3]
        return params, grads


# Main function to run the experiments
def main():
    data = load_data()
    batch_size = 128
    input_size = 64  # For the digits dataset (8x8 images flattened)
    hidden_size = 20
    output_size = 10
    epochs = 200

    print("Training with SGD LR 0.5")
    network = NeuralNetwork(input_size, hidden_size, hidden_size, output_size)
    test_losses_sgd, test_accuracies_sgd = train(network, data, SGD(
        network.get_params_and_grads()[0], 0.5), epochs, batch_size)

    print("Training with SGD LR 0.2")
    network = NeuralNetwork(input_size, hidden_size, hidden_size, output_size)
    test_losses_sgd_b, test_accuracies_sgd_b = train(network, data, SGD(
        network.get_params_and_grads()[0], 0.2), epochs, batch_size)

    print("Training with SGD with Momentum LR 0.05")
    network = NeuralNetwork(input_size, hidden_size, hidden_size, output_size)
    test_losses_momentum, test_accuracies_momentum = train(network, data, SGD_Momentum(
        network.get_params_and_grads()[0], 0.05, 0.9), epochs, batch_size)

    print("Training with AdaGrad LR 0.05")
    network = NeuralNetwork(input_size, hidden_size, hidden_size, output_size)
    test_losses_adagrad, test_accuracies_adagrad = train(network, data, SGD_AdaGrad(
        network.get_params_and_grads()[0], 0.05, 1e-8), epochs, batch_size)

    print("Training with AdaGrad LR 0.01")
    network = NeuralNetwork(input_size, hidden_size, hidden_size, output_size)
    test_losses_adagrad_b, test_accuracies_adagrad_b = train(network, data, SGD_AdaGrad(
        network.get_params_and_grads()[0], 0.01, 1e-8), epochs, batch_size)

    # Compare train losses and train accuracies
    all_losses = [test_losses_sgd, test_losses_sgd_b,
                  test_losses_momentum, test_losses_adagrad, test_losses_adagrad_b]
    all_accuracies = [test_accuracies_sgd, test_accuracies_sgd_b,
                      test_accuracies_momentum, test_accuracies_adagrad, test_accuracies_adagrad_b]
    all_labels = ["SGD LR 0.5", "SGD LR 0.2",
                  "SGD-Momentum LR 0.05", "AdaGrad LR 0.05", "AdaGrad LR 0.01"]
    plot_all_results(all_losses, all_accuracies,
                     all_labels, filename="3layers.png")


if __name__ == "__main__":
    main()
