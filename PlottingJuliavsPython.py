import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# LOAD DATA
# =========================
metrics_py = pd.read_csv("python_performance_metrics.csv")
metrics_jl = pd.read_csv("performance_metrics.csv")

# =========================
# LOSS PLOTS (PER RUN)
# =========================
plt.figure(figsize=(10,6))

# Python
plt.plot(metrics_py["loss_const"], label="Python Constant", color= "lightblue" )
plt.plot(metrics_py["loss_lin"], label="Python Linear", color = "deepskyblue")
plt.plot(metrics_py["loss_NN"], label="Python NN",color = "dodgerblue")

# Julia
plt.plot(metrics_jl["loss_const"], label="Julia Constant", color="fuchsia")
plt.plot(metrics_jl["loss_lin"], label="Julia Linear", color = "hotpink")
plt.plot(metrics_jl["loss_NN"], label="Julia NN", color ="mediumvioletred")

plt.xlabel("Run")
plt.ylabel("Loss")
plt.title("Loss Comparison per Run (Python vs Julia)")
plt.legend()
plt.grid()
plt.show()


# =========================
# VALIDATION LOSS PLOT
# =========================
plt.figure(figsize=(10,6))

plt.plot(metrics_py["loss_const_val"], label="Python Const Val", color= "lightblue")
plt.plot(metrics_py["loss_lin_val"], label="Python Lin Val", color = "deepskyblue")
plt.plot(metrics_py["loss_NN_val"], label="Python NN Val", color = "dodgerblue")

plt.plot(metrics_jl["loss_const_val"], label="Julia Const Val",color="fuchsia")
plt.plot(metrics_jl["loss_lin_val"], label="Julia Lin Val",color = "hotpink")
plt.plot(metrics_jl["loss_NN_val"], label="Julia NN Val",color ="mediumvioletred")

plt.xlabel("Run")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid()
plt.show()


# =========================
# MSE BAR PLOT (MEAN)
# =========================

labels = ["MSE11", "MSE12", "MSE13"]

py_means = [
    metrics_py["MSE11_NN_val"].mean(),
    metrics_py["MSE12_NN_val"].mean(),
    metrics_py["MSE13_NN_val"].mean()
]

jl_means = [
    metrics_jl["MSE11_NN_val"].mean(),
    metrics_jl["MSE12_NN_val"].mean(),
    metrics_jl["MSE13_NN_val"].mean()
]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, py_means, width, label="Python", color = "mediumblue")
plt.bar(x + width/2, jl_means, width, label="Julia", color = "hotpink")

plt.xticks(x, labels)
plt.ylabel("Mean MSE")
plt.title("MSE Comparison (NN Model)")
plt.legend()
plt.grid(axis='y')
plt.show()


# =========================
# OPTIONAL: ALL MODELS MSE COMPARISON
# =========================

def get_means(df, prefix):
    return [
        df[f"MSE11_{prefix}_val"].mean(),
        df[f"MSE12_{prefix}_val"].mean(),
        df[f"MSE13_{prefix}_val"].mean()
    ]

models = ["const", "lin", "NN"]

for m in models:
    py = get_means(metrics_py, m)
    jl = get_means(metrics_jl, m)

    plt.figure(figsize=(8,5))
    plt.bar(x - width/2, py, width, label="Python", color = "mediumblue")
    plt.bar(x + width/2, jl, width, label="Julia", color = "hotpink")

    plt.xticks(x, labels)
    plt.ylabel("Mean MSE")
    plt.title(f"MSE Comparison ({m.upper()} Model)")
    plt.legend()
    plt.grid(axis='y')
    plt.show()