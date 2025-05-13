import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stats.csv")

df["BlockSize"] = df["ThreadsX"].astype(str) + "x" + df["ThreadsY"].astype(str)

df = df[["BlockSize", "MatrixSize", "Time(mcs)"]]

agg_df = df.groupby(["BlockSize", "MatrixSize"]).agg(
    mean_time=("Time(mcs)", "mean"),
    std_time=("Time(mcs)", "std")
).reset_index()

plt.figure(figsize=(10, 6))
for block_size, group in agg_df.groupby("BlockSize"):
    group = group.sort_values("MatrixSize")
    plt.errorbar(
        group["MatrixSize"],
        group["mean_time"],
        yerr=group["std_time"],
        label=f"{block_size} Block",
        marker='o',
        capsize=5
    )

plt.title("Зависимость времени от размера матрицы для разных CUDA блоков")
plt.xlabel("Matrix Size")
plt.ylabel("Time (mcs)")
plt.legend(title="CUDA Block Size")
plt.grid(True)
plt.tight_layout()
plt.savefig("res.png", dpi=300)
plt.show()
