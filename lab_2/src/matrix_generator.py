import numpy as np


def matrix_generate(matrix: dict):
    new_matrix = np.random.randint(
        matrix["min"], matrix["max"], size=(matrix["rows"], matrix["cols"]))
    with open(f"{matrix["path"]}", "w") as file:
        for row in new_matrix:
            file.write(" ".join(map(str, row)) + "\n")
