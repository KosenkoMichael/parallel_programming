import os
import numpy as np


result_dir = "result"

if not os.path.isdir(result_dir):
    print(f"папка {result_dir} не найдена")
    exit(1)

for subfolder in os.listdir(result_dir):
    subfolder_path = os.path.join(result_dir, subfolder)
    if os.path.isdir(subfolder_path):
        try:
            path_a = os.path.join(subfolder_path, "A.txt")
            path_b = os.path.join(subfolder_path, "B.txt")
            path_c = os.path.join(subfolder_path, "C.txt")

            A = np.loadtxt(path_a)
            B = np.loadtxt(path_b)
            C = np.loadtxt(path_c)

            true_result = np.dot(A, B)

            if np.allclose(true_result, C):
                print(f"for {subfolder_path} matrix multiplication is correct")
            else:
                print(f"for {subfolder_path} matrix multiplication is NOT correct")
        except Exception as e:
            print(f"[{subfolder}] Error: {e}")
