import os
import subprocess
import numpy as np
from src import matrix_generator

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)

matrix_1 = {
    "min": 0,
    "max": 10,
    "rows": 3,
    "cols": 4,
    "path": "lab_1/src/matrix/1.txt",
}

matrix_2 = {
    "min": 0,
    "max": 10,
    "rows": 4,
    "cols": 3,
    "path": "lab_1/src/matrix/2.txt",
}

matrix_generator.matrix_generate(matrix_1)
matrix_generator.matrix_generate(matrix_2)

mat1 = np.loadtxt(matrix_1["path"])
mat2 = np.loadtxt(matrix_2["path"])

right_result = mat1 @ mat2
