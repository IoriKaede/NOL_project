import numpy as np

from supplementary import Value


def weight_init_function(layer_size1: int, layer_size2: int):
    # return np.random.uniform(-1, 1, (layer_size1, layer_size2))
    return np.random.randn(layer_size1, layer_size2) * np.sqrt(2.0 / layer_size1)

def bias_init_function(layer_size: int):
    # return np.random.uniform(-1, 1, layer_size)
    return np.zeros(layer_size)

class NeuralNetwork:
    r"""Neural network class.
    """
    def __init__(self, layers, activation_functions):
        self.number_of_layers = len(layers) - 1
        self.biases = []
        self.weights = []
        self.activation_functions = activation_functions

        if len(activation_functions) != self.number_of_layers:
            raise ValueError("Number of activation functions should match the number of layers.")

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size), expr=f"$b^{{{i}}}$")
                )
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(weight_init_function(layer_size1=size, layer_size2=layers[i+1]), expr=f"$(W^T)^{{{i}}}$")
                )

        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        self.c_weights = [np.zeros_like(w.data) for w in self.weights]
        self.v_weights = [np.zeros_like(w.data) for w in self.weights]

        self.c_biases = [np.zeros_like(b.data) for b in self.biases]
        self.v_biases = [np.zeros_like(b.data) for b in self.biases]

    def __call__(self, x):
        for (weight, bias, activation_function) in zip(self.weights, self.biases, self.activation_functions):
            x = activation_function(x @ weight + bias)
        return x

    def adam_step(self, learning_rate):
        self.t += 1
        for i, weight in enumerate(self.weights):
            grad = weight.grad

            self.v_weights[i] = (self.beta1 * self.v_weights[i] + (1 - self.beta1) * grad)
            self.c_weights[i] = (self.beta2 * self.c_weights[i] + (1 - self.beta2) * (grad ** 2))

            v_tilde = self.v_weights[i] / (1 - self.beta1 ** self.t)
            c_tilde = self.c_weights[i] / (1 - self.beta2 ** self.t)

            weight.data -= learning_rate * v_tilde / (np.sqrt(self.eps + c_tilde))

        for i, bias in enumerate(self.biases):
            grad = bias.grad

            self.v_biases[i] = (self.beta1 * self.v_biases[i] + (1 - self.beta1) * grad)
            self.c_biases[i] = (self.beta2 * self.c_biases[i] + (1 - self.beta2) * (grad ** 2))

            v_tilde = self.v_biases[i] / (1 - self.beta1 ** self.t)
            c_tilde = self.c_biases[i] / (1 - self.beta2 ** self.t)

            bias.data -= learning_rate * v_tilde / (np.sqrt(self.eps + c_tilde))

    def reset_gradients(self):
        for weight in self.weights:
            weight.reset_grad()
        for bias in self.biases:
            bias.reset_grad()

    def gradient_descent(self, learning_rate):
        for weight in self.weights:
            weight.data -= learning_rate * weight.grad
        for bias in self.biases:
            bias.data -= learning_rate * bias.grad

    def save(self, path):
        np_weights = [weight.data for weight in self.weights]
        np_biases = [bias.data for bias in self.biases]

        np.savez(path / "weights.npz", *np_weights)
        np.savez(path / "biases.npz", *np_biases)

    def load(self, path):
        np_weights = [arr for arr in np.load(path / "weights.npz").values()]
        np_biases = [arr for arr in np.load(path / "biases.npz").values()]

        self.weights = [
            Value(weight, expr=f"$W^{{{i}}}$") for i, weight in enumerate(np_weights)
        ]
        self.biases = [
            Value(bias, expr=f"$b^{{{i}}}$") for i, bias in enumerate(np_biases)
        ]
