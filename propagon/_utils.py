from typing import List
from propagon._err import MatrixDimensionError
import math

tanh = lambda x: math.tanh(x)
d_tanh = lambda output: 1 - output ** 2

sigmoid = lambda x: 1 / (1 + math.exp(-x))
d_sigmoid = lambda output: output * (1 - output)

relu = lambda x: max(0, x)
d_relu = lambda output: 1 if output > 0 else 0

leaky_relu = lambda x: max(0.01 * x, x)
d_leaky_relu = lambda output: 1 if output > 0 else 0.01

acts = {
    "tanh": {
        "f": tanh,
        "d": d_tanh
    },
    "sigmoid": {
        "f": sigmoid,
        "d": d_sigmoid
    },
    "relu": {
        "f": relu,
        "d": d_relu
    },
    "leaky_relu": {
        "f": leaky_relu,
        "d": d_leaky_relu
    }
}

def mse(y: List[List[float]], y_hat: List[List[float]]) -> float:
    return sum([Y - YH for Y, YH in zip(y[0], y_hat[0])])

def dot(a: List[List[float]], b: List[List[float]], act: str = None, bias: float = 0) -> List[List[float]]:
    def is_valid() -> None:
        if not is_rectangular(a): 
            raise MatrixDimensionError("First matrix is not rectangular")
        if not is_rectangular(b):
            raise MatrixDimensionError("Second matrix is not rectangular")
        if len(a[0]) != len(b): 
            raise MatrixDimensionError("Matrix dimensions do not match")
        if act is not None and act not in acts:
            raise ValueError("Invalid Activation function")

    def activate(x: float) -> float:
        return x if act is None else acts[act]["f"](x)

    is_valid()

    try:
        return [
            [
                activate(
                    sum(a[i][k] * b[k][j] for k in range(len(a[0])))
                    + bias
                )
                for j in range(len(b[0]))
            ]
            for i in range(len(a))
        ]

    except Exception as e:
        raise ValueError("Matrix multiplication failed") from e

def sub(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise MatrixDimensionError("Matrix dimensions do not match")
    return [
        [a[i][j] - b[i][j] for j in range(len(a[0]))]
        for i in range(len(a))
    ]

def add(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise MatrixDimensionError("Matrix dimensions do not match")
    return [
        [a[i][j] + b[i][j] for j in range(len(a[0]))]
        for i in range(len(a))
    ]

def is_rectangular(a) -> bool:
    row_lengths = [len(row) for row in a]
    return all(length == row_lengths[0] for length in row_lengths)

def transpose(a: List[List[float]]) -> List[List[float]]:
    if not is_rectangular(a):
        raise ValueError("Matrix is not rectangular")
    return [
        [a[j][i] for j in range(len(a))]
        for i in range(len(a[0]))
    ]

def hadamard(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise MatrixDimensionError(f"Matrix dimensions do not match: {dim(a)} and {dim(b)}")
    return [
        [x * y for x, y in zip(row_1, row_2)]
        for row_1, row_2 in zip(a, b)
    ]

"""
def hadamard(*matrices: List[List[float]]) -> List[List[float]]:
    if (len(set(len(matrix) for matrix in matrices)) > 1 or len(set(len(row) for row in matrix) for matrix in matrices) > 1):
        raise MatrixDimensionError("Matrix dimensions do not match")
    return [
        [math.prod(elements) for elements in zip(*rows)]
        for rows in zip(*matrices)
    ]
"""

def scale(a: List[List[float]], scalar: float) -> List[List[float]]:
    return [
        [x * scalar for x in row]
        for row in a
    ]

def dim(a: List[List[float]]) -> str:
    return f"{len(a)}x{len(a[0])}"
