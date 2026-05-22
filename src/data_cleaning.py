import pandas as pd

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Check missing values
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

print("Data cleaned successfully!")