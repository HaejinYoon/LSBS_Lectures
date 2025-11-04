import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
n = np.array([61, 85, 75, 86, 64, 96, 87, 93, 67, 97, 77, 88, 90, 84, 65, 71, 69, 66, 98, 72])
x_i = np.array([4, 3, 2, 4, 2, 4, 5, 3, 6, 7, 6, 5, 8, 5, 5, 3, 3, 4, 5, 8])
p_hat = np.sum(x_i) / np.sum(n)
p_hat

ucl = p_hat + 3 * np.sqrt(p_hat * (1 - p_hat) / n)
lcl = p_hat - 3 * np.sqrt(p_hat * (1 - p_hat) / n)

day_i = np.arange(1, 21)
p_i = x_i / n
df = pd.DataFrame({
    "Day": day_i,
    "Defective Rate": p_i,
    "UCL": ucl,
    "LCL": lcl,
    "Ave. Rate": [p_hat] * 20
})
sns.lineplot(x="Day", y="Defective Rate", data=df, marker="o", label="Defective Rate")
sns.lineplot(x="Day", y="UCL", data=df, color='red', label="UCL")
sns.lineplot(x="Day", y="LCL", data=df, color='red', label="LCL")
sns.lineplot(x="Day", y="Ave. Rate", data=df, color='black', linestyle='--', label="Ave. Rate")
plt.fill_between(df["Day"], df["LCL"], df["UCL"], color='red', alpha=0.1)
plt.title('P chart')
plt.ylabel('Defective Rate')
plt.legend(loc="lower right")
plt.show()