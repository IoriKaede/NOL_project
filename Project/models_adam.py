# models.py
import numpy as np

from supplementary import Value


def weight_init_function(layer_size1: int, layer_size2: int):
    # Kaiming-He Initialization (Normal Distribution)
    std = np.sqrt(2.0 / layer_size1)
    return np.random.normal(0, std, (layer_size1, layer_size2))


def he_weight_init(layer_size1: int, layer_size2: int):
    std = np.sqrt(2.0 / layer_size1)
    return np.random.randn(layer_size1, layer_size2) * std


def bias_init_function(layer_size: int):

    return np.zeros(layer_size)


class NeuralNetwork:
    r"""Neural network class."""

    def __init__(self, layers, activation_functions):
        self.number_of_layers = len(layers) - 1
        self.biases = []
        self.weights = []
        self.activation_functions = activation_functions

        if len(activation_functions) != self.number_of_layers:
            raise ValueError(
                "Number of activation functions should match the number of layers."
            )

        for i, size in enumerate(layers):
            if i > 0:
                self.biases.append(
                    Value(data=bias_init_function(layer_size=size), expr=f"$b^{{{i}}}$")
                )
            if i < self.number_of_layers:
                # We initialize the weights transposed
                self.weights.append(
                    Value(
                        he_weight_init(layer_size1=size, layer_size2=layers[i + 1]),
                        expr=f"$(W^T)^{{{i}}}$",
                    )
                )

        self.t = 0

        self.v_weights = [np.zeros_like(w.data) for w in self.weights]
        self.c_weights = [np.zeros_like(w.data) for w in self.weights]

        self.v_biases = [np.zeros_like(b.data) for b in self.biases]
        self.c_biases = [np.zeros_like(b.data) for b in self.biases]

    def __call__(self, x):
        for weight, bias, activation_function in zip(
            self.weights, self.biases, self.activation_functions
        ):
            x = activation_function(x @ weight + bias)
        return x

    def adam_step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        beta1_t = 1 - beta1**self.t
        beta2_t = 1 - beta2**self.t
        for i, weight in enumerate(self.weights):
            g = weight.grad

            v = self.v_weights[i]
            c = self.c_weights[i]

            v *= beta1
            v += (1 - beta1) * g

            c *= beta2
            c += (1 - beta2) * (g * g)

            weight.data -= lr * (v / beta1_t) / (np.sqrt(c / beta2_t) + eps)

        for i, bias in enumerate(self.biases):
            g = bias.grad

            v = self.v_biases[i]
            c = self.c_biases[i]

            v *= beta1
            v += (1 - beta1) * g

            c *= beta2
            c += (1 - beta2) * (g * g)

            bias.data -= lr * (v / beta1_t) / (np.sqrt(c / beta2_t) + eps)

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
