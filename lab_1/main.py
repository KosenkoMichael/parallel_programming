import os
import subprocess
import numpy as np
from src import matrix_generator

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
mat_dir_path = "src/matrix/"

matrix_1 = {
    "min": 0,
    "max": 10,
    "rows": 3,
    "cols": 4,
    "path": f"{mat_dir_path}1.txt",
}

matrix_2 = {
    "min": 0,
    "max": 10,
    "rows": 4,
    "cols": 3,
    "path": f"{mat_dir_path}2.txt",
}

matrix_generator.matrix_generate(matrix_1)
matrix_generator.matrix_generate(matrix_2)

mat1 = np.loadtxt(matrix_1["path"])
mat2 = np.loadtxt(matrix_2["path"])

right_result = mat1 @ mat2

exe_path = r"build/matrix.exe"
arguments = [matrix_1["path"], matrix_2["path"], f"{mat_dir_path}result.txt"]
result = subprocess.run([exe_path] + arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

return_code = result.returncode
print(f"Код возврата: {return_code}")

mat_cpp = np.loadtxt(f"{mat_dir_path}result.txt")

if(np.array_equal(mat_cpp, right_result)):
    print("the multiplication was performed correctly")