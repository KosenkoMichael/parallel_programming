import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from src import matrix_generator

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
mat_dir_path = "src/matrix/"

matrix_1 = {
    "min": 0,
    "max": 10,
    "rows": 100,
    "cols": 100,
    "path": f"{mat_dir_path}1.txt",
}

matrix_2 = {
    "min": 0,
    "max": 10,
    "rows": 100,
    "cols": 100,
    "path": f"{mat_dir_path}2.txt",
}

mat_size = []
mat_time = []

iterations_num = int(sys.argv[1]) if len(sys.argv) == 3 else 10
for i in range(iterations_num):
    matrix_generator.matrix_generate(matrix_1)
    matrix_generator.matrix_generate(matrix_2)

    mat1 = np.loadtxt(matrix_1["path"])
    mat2 = np.loadtxt(matrix_2["path"])

    right_result = mat1 @ mat2

    exe_path = r"build/matrix.exe"
    arguments = [matrix_1["path"], matrix_2["path"],
                 f"{mat_dir_path}result.txt"]
    result = subprocess.run([exe_path] + arguments,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return_code = result.returncode

    mat_size.append(matrix_1["rows"] * matrix_1["cols"])
    mat_time.append(return_code)

    mat_cpp = np.loadtxt(f"{mat_dir_path}result.txt")

    if (np.array_equal(mat_cpp, right_result)):
        print(f"№{i})the multiplication was performed correctly")
    else:
        raise ValueError("Matrix doesn't match")
    matrix_1["rows"] += int(sys.argv[2])
    matrix_1["cols"] += int(sys.argv[2])
    matrix_2["rows"] += int(sys.argv[2])
    matrix_2["cols"] += int(sys.argv[2])

plt.plot(mat_size, mat_time, marker='o', linestyle='-', color='b')
plt.title("size-time dependance")
plt.xlabel("Matrix size")
plt.ylabel("Multiplication time") 
plt.show()