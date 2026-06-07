import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/student_performance.csv")

print(df.shape)

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df["study_hours"], bins=10)

plt.title("Study Hours Distribution")

plt.savefig("images/histogram.png")
plt.close()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Heatmap")

plt.savefig("images/heatmap.png")
plt.close()

# Scatter Plot
plt.figure(figsize=(8,5))

sns.scatterplot(
    x=df["study_hours"],
    y=df["final_score"]
)

plt.title("Study Hours vs Final Score")

plt.savefig("images/scatter_plot.png")
plt.close()

print("EDA graphs saved successfully!")