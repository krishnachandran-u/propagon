import random
import sys
from propagon import Layer, NeuralNet

if __name__ == "__main__":
    x = [[random.uniform(-1, 1) for _ in range(5)] for _ in range(100)]
    y = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(100)]

    layers = [
        Layer(5, act="relu"),
        Layer(4, act="relu"),
        Layer(3, act="relu"),
        Layer(3, act="sigmoid")
    ]
    
    neuralnet = NeuralNet(layers)

    x_train, y_train = x[:80], y[:80] 
    x_test, y_test = x[80:], y[80:]

    neuralnet.train(x_train, y_train, alpha=0.01, epochs=100)

    y_pred, err = neuralnet.predict(x_test)

    print(f"Predictions: {y_pred}")
    print(f"Mean Squared Error: {err}")