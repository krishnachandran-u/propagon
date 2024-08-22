from typing import List
import random
from _utils import dot
class Layer:
    n_neurons: int
    prev_layer: 'Layer' 
    next_layer: 'Layer'

    def __init__(self, n_neurons: int):
        self.n_neurons = n_neurons

class NeuralNet:
    layers: List[Layer]
    weights: List[List[List[float]]]

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self._init_weights()

    def _init_weights_2d(self, m: int, n: int) -> List[List[float]]: # make an n x m matrix - m: neurons in previous layer, n: neurons in next layer
        weights_2d = [[0]*m for _ in range(n)]
        for j in range(n):
            for k in range(m):
                weights_2d[j][k] = random.uniform(-1, 1)
        return weights_2d

    def _init_weights(self) -> None:
        for i in range(len(self.layers) - 1):
            self.weights.append(self._init_weights_2d(self.layers[i].n_neurons, self.layers[i + 1].n_neurons))

    def _feed_forward(self, x: List[float]) -> List[float]: 
        outputs = x
        for i in range(len(self.layers) - 1):
            outputs = dot(outputs, self.weights[i])
        return outputs

    def train(self, x: List[float], y: List[float], learning_rate: float):
        if len(x) != self.layers[0].n_neurons:
            raise ValueError("Input layer size mismatch")
        if len(y) != self.layers[-1].n_neurons:
            raise ValueError("Output layer size mismatch")

        # run a loop with feed forward and backpropagation 