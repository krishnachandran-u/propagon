from typing import List
import random
from propagon._utils import *
class Layer:
    n_neurons: int
    bias: float
    act: str

    def __init__(self, n_neurons: int, bias: float = 0, act: str = None):
        self.n_neurons = n_neurons
        self.bias = bias
        if act is not None and act not in acts:
            raise ValueError("Invalid Activation function")
        self.act = act

class NeuralNet:
    layers: List[Layer]
    weights: List[List[List[float]]]

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._init_weights()

    def _init_weights_2d(self, m: int, n: int) -> List[List[float]]: #im not sure if this is correct
        weights_2d = [[0]*n for _ in range(m)]
        for j in range(m):
            for k in range(n):
                weights_2d[j][k] = random.uniform(-1, 1)
        return weights_2d

    def _init_weights(self) -> None:
        self.weights = []
        for i in range(len(self.layers) - 1):
            self.weights.append(self._init_weights_2d(self.layers[i].n_neurons, self.layers[i + 1].n_neurons))

    def train(self, x: List[List[float]], y: List[List[float]], alpha: float, epochs: int) -> None:
        if len(x) != len(y):
            raise ValueError("Input and output count mismatch")
        if not is_rectangular(x):
            raise ValueError("Input matrix is not rectangular")
        if not is_rectangular(y):
            raise ValueError("Output matrix is not rectangular")

        for i in range(len(x)):
            for _ in range(epochs):
                A = [[x[i]]]
                A = self.feed_forward(A)
                D = self.back_propagation(A, [y[i]])
                self.update_weights(A, D, alpha)

    def feed_forward(self, A: List[List[List[float]]]) -> List[List[List[float]]]:
        for i in range(len(self.layers) - 1):
            A.append(
                dot(A[i], self.weights[i], self.layers[i + 1].act, self.layers[i + 1].bias)
            )
        return A

    def back_propagation(self, A: List[List[List[float]]], y: List[List[float]]) -> List[List[List[float]]]:
        err = sub(A[-1], y)
        err = [
            [x**2 for x in row]
            for row in err
        ]
        D = [hadamard(err, [[acts[self.layers[-1].act]["d"](x) for x in A[-1][0]]])]

        for layer in range(len(self.layers) - 2, 0, -1):
            delta = dot(D[0], transpose(self.weights[layer]))   
            delta = hadamard(delta, [[acts[self.layers[layer].act]["d"](x) for x in A[layer][0]]])
            D.insert(0, delta)

        return D

    def update_weights(self, A: List[List[float]], D: List[List[float]], alpha: float) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = sub(self.weights[i], scale(dot(transpose(A[i]), D[i]), alpha))

    def predict(self, x: List[List[float]], y: List[List[float]] = None) -> List[List[float]]:
        if not is_rectangular(x):
            raise ValueError("Input matrix is not rectangular")
        y_hat = []
        for i in range(len(x)):
            A = [[x[i]]]
            y_hat.append(self.feed_forward(A)[-1])

        err = None
        if y is not None:
            err = mse(y, y_hat)

        return y_hat, err