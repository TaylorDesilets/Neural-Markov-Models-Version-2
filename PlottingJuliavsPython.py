import pandas as pd
import matplotlib.pyplot as plt

# Load data
metrics_py = pd.read_csv("metrics_python.csv")
metrics_jl = pd.read_csv("metrics_julia.csv")

x = range(len(metrics_py["model"]))

# -------------------------
# MSE_12 COMPARISON
# -------------------------
plt.figure(figsize=(8,5))

plt.bar([i - 0.2 for i in x], metrics_py["mse_12"],
        width=0.4, label="Python", alpha=0.7, edgecolor='black')

plt.bar([i + 0.2 for i in x], metrics_jl["mse_12"],
        width=0.4, label="Julia", alpha=0.7, edgecolor='black')

plt.xticks(x, metrics_py["model"])
plt.title("MSE 12 Comparison")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()


# -------------------------
# MSE_13 COMPARISON
# -------------------------
plt.figure(figsize=(8,5))

plt.bar([i - 0.2 for i in x], metrics_py["mse_13"],
        width=0.4, label="Python", alpha=0.7, edgecolor='black')

plt.bar([i + 0.2 for i in x], metrics_jl["mse_13"],
        width=0.4, label="Julia", alpha=0.7, edgecolor='black')

plt.xticks(x, metrics_py["model"])
plt.title("MSE 13 Comparison")
plt.ylabel("MSE")
plt.yscale("log")
plt.legend()

plt.tight_layout()
plt.show()


# -------------------------
# FINAL LOSS COMPARISON  ✅ NEW
# -------------------------
plt.figure(figsize=(8,5))

plt.bar([i - 0.2 for i in x], metrics_py["final_loss"],
        width=0.4, label="Python", alpha=0.7, edgecolor='black')

plt.bar([i + 0.2 for i in x], metrics_jl["final_loss"],
        width=0.4, label="Julia", alpha=0.7, edgecolor='black')

plt.xticks(x, metrics_py["model"])
plt.title("Final Loss Comparison")
plt.ylabel("Loss")

plt.yscale("log")

plt.legend()

plt.tight_layout()
plt.show()


# Debug print (keep this)
print(metrics_py)
print(metrics_jl)