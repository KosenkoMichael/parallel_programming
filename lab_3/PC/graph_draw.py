import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stats.csv")

df = df[["Threads", "MatrixSize", "Time(ms)"]]

agg_df = df.groupby(["Threads", "MatrixSize"]).agg(
    mean_time=("Time(ms)", "mean"),
    std_time=("Time(ms)", "std")
).reset_index()

grouped = agg_df.groupby("Threads")

plt.figure(figsize=(10, 6))

for thread_count, group in grouped:
    group = group.sort_values("MatrixSize")
    
    plt.errorbar(
        group["MatrixSize"],
        group["mean_time"],
        yerr=group["std_time"],
        label=f"{thread_count} Threads",
        marker='o',
        capsize=5
    )

plt.title("Зависимость времени от размера матрицы")
plt.xlabel("Matrix Size")
plt.ylabel("Time(ms)")
plt.legend(title="Threads")
plt.grid(True)
plt.tight_layout()

plt.savefig("res.png", dpi=300)

plt.show()
