from typing import List
import random
from utils import get_random_weight

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
        for i in range(len(layers) - 1):
            layers[i].next_layer = layers[i+1]
            layers[i+1].prev_layer = layers[i]
        layers[0].prev_layer = None
        layers[-1].next_layer = None
        
        self.weights = []
        for i in range(len(layers) - 1):
            self.weights.append([get_random_weight()] * layers[i].n_neurons * layers[i+1].n_neurons)



    