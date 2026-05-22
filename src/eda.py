import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['study_hours'], bins=10)

plt.title("Study Hours Distribution")
plt.xlabel("Study Hours")
plt.ylabel("Students")

plt.savefig("images/histogram.png")
plt.close()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.savefig("images/heatmap.png")
plt.close()

print("EDA graphs saved successfully!")