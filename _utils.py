from typing import List
from _err import MatrixDimensionError

def dot(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    def is_valid(a: List[List[int]]) -> bool:
        row_lengths_a = [len(row) for row in a]
        if not all(length == row_lengths_a[0] for length in row_lengths_a): 
            raise MatrixDimensionError("First matrix is not rectangular")
        row_lengths_b = [len(row) for row in b]
        if not all(length == row_lengths_b[0] for length in row_lengths_b):
            raise MatrixDimensionError("Second matrix is not rectangular")
        if len(a[0]) != len(b): 
            raise MatrixDimensionError("Matrix dimensions do not match")

    is_valid()

    try:
        return [[sum(a[i][k] * b[k][j] for k in range(len(a[0]))) for j in range(len(b[0]))] for i in range(len(a))]
    except Exception as e:
        raise ValueError("Matrix multiplication failed") from e