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
    "rows": 2,
    "cols": 2,
    "path": f"{mat_dir_path}1.txt",
}

matrix_2 = {
    "min": 0,
    "max": 10,
    "rows": 2,
    "cols": 2,
    "path": f"{mat_dir_path}2.txt",
}

mat_size = []
mat_time = []

iterations_num = int(sys.argv[1]) if len(sys.argv) == 4 else 10
size_grow = int(sys.argv[2]) if len(sys.argv) == 4 else 50
repeat_num = int(sys.argv[3]) if len(sys.argv) == 4 else 5

for i in range(iterations_num):
    for j in range(repeat_num):
        matrix_generator.matrix_generate(matrix_1)
        matrix_generator.matrix_generate(matrix_2)

        mat1 = np.loadtxt(matrix_1["path"])
        mat2 = np.loadtxt(matrix_2["path"])

        right_result = mat1 @ mat2

        exe_path = r"lab_2_OpenMP/x64\Debug/lab_2_OpenMP.exe"
        arguments = [matrix_1["path"], matrix_2["path"],
                    f"{mat_dir_path}result.txt"]
        result = subprocess.run([exe_path] + arguments,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return_code = result.returncode

        mat_size.append(matrix_1["rows"])
        mat_time.append(return_code)

        mat_cpp = np.loadtxt(f"{mat_dir_path}result.txt")

        if (np.array_equal(mat_cpp, right_result)):
            print(f"№{i},{j})the multiplication was performed correctly")
        else:
            raise ValueError("Matrix doesn't match")
    matrix_1["rows"] += size_grow
    matrix_1["cols"] += size_grow
    matrix_2["rows"] += size_grow
    matrix_2["cols"] += size_grow

plt.plot(mat_size, mat_time, marker='o', linestyle='none', color='b')
plt.title("size-time dependance")
plt.xlabel("Matrix size")
plt.ylabel("Multiplication time") 
plt.show()