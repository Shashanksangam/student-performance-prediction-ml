import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================
# Load Dataset
# ======================

df = pd.read_csv("data/student_performance.csv")

print(f"\nDataset Shape: {df.shape}")

# ======================
# Features and Target
# ======================

X = df.drop("final_score", axis=1)
y = df["final_score"]

# ======================
# Train-Test Split
# ======================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

# ======================
# Linear Regression
# ======================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# ======================
# Random Forest
# ======================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# ======================
# Evaluation Function
# ======================

def evaluate_model(model_name, y_true, predictions):

    mae = mean_absolute_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)

    print("\n" + "=" * 40)
    print(model_name)
    print("=" * 40)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

# ======================
# Results
# ======================

evaluate_model(
    "Linear Regression",
    y_test,
    lr_predictions
)

evaluate_model(
    "Random Forest Regression",
    y_test,
    rf_predictions
)



# =====================================
# Predict New Students
# =====================================

new_students = pd.DataFrame([
    [8, 90, 7, 85, 9, 3, 88],
    [4, 75, 6, 68, 5, 6, 70]
], columns=[
    "study_hours",
    "attendance",
    "sleep_hours",
    "previous_grades",
    "assignments_completed",
    "internet_usage",
    "test_scores"
])

predictions = lr_model.predict(new_students)

print("\nPredictions for New Students:")

for i, score in enumerate(predictions, start=1):
    print(f"Student {i}: Predicted Final Score = {score:.2f}")