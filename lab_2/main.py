import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
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

res = {}

iterations_num = int(sys.argv[1])
repeat_count = int(sys.argv[3])
omp_set_num_threads = sys.argv[4]
for i in range(iterations_num):
    interim_res = []
    for j in range(repeat_count):
        matrix_generator.matrix_generate(matrix_1)
        matrix_generator.matrix_generate(matrix_2)

        mat1 = np.loadtxt(matrix_1["path"])
        mat2 = np.loadtxt(matrix_2["path"])

        right_result = mat1 @ mat2

        exe_path = r"lab_2_OpenMP/x64/Debug/lab_2_OpenMP.exe"
        arguments = [matrix_1["path"], matrix_2["path"],
                     f"{mat_dir_path}result.txt", omp_set_num_threads]
        result = subprocess.run([exe_path] + arguments,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        return_code = result.returncode

        interim_res.append(return_code)

        mat_cpp = np.loadtxt(f"{mat_dir_path}result.txt")

        if np.array_equal(mat_cpp, right_result):
            print(f"№{i},{j}) the multiplication was performed correctly")
        else:
            raise ValueError("Matrix doesn't match")

    res[matrix_1["rows"]] = interim_res
    matrix_1["rows"] += int(sys.argv[2])
    matrix_1["cols"] += int(sys.argv[2])
    matrix_2["rows"] += int(sys.argv[2])
    matrix_2["cols"] += int(sys.argv[2])

sizes = sorted(res.keys())
means = [np.mean(res[size]) for size in sizes]
std_devs = [np.std(res[size], ddof=1) for size in sizes]
conf_margin = [stats.t.ppf(0.975, len(res[size]) - 1) * (std / np.sqrt(len(res[size])))
               for size, std in zip(sizes, std_devs)]

conf_margin = [round(m, 3) for m in conf_margin]

conf_lower = [round(mean - margin, 3) for mean, margin in zip(means, conf_margin)]
conf_upper = [round(mean + margin, 3) for mean, margin in zip(means, conf_margin)]

df = pd.DataFrame({
    "Matrix Size": sizes,
    "Execution Times": [",".join(map(str, res[size])) for size in sizes],
    "Confidence Interval Lower": conf_lower,
    "Confidence Interval Upper": conf_upper
})

csv_filename = "matrix_multiplication_results.csv"
df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

plt.errorbar(sizes, means, yerr=conf_margin, fmt='o', color='b', ecolor='r', capsize=5, label="Mean with 95% CI")
plt.title("Size-Time Dependence with Confidence Intervals")
plt.xlabel("Matrix Size")
plt.ylabel("Multiplication Time, milliseconds")
plt.legend()
plt.grid(True)
plt.savefig("image.png")
plt.show()
