import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Step 2: Load dataset
data = pd.read_csv("Dataset/student-scores.csv")

print("First 5 rows of dataset:")
print(data.head())

# Step 3: Basic info
print("\nDataset Info:")
print(data.info())

print("\nChecking for missing values:")
print(data.isnull().sum())

# Step 4: Focused Data Visualization
sns.pairplot(
    data[["weekly_self_study_hours", "absence_days", "math_score", "english_score"]],
    diag_kind="kde"
)
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# Step 5: Feature selection (study_hours → final_score)
# Use weekly study hours as independent variable
X = data[['weekly_self_study_hours']]

# Define ExamScore as average of all subject scores
data['ExamScore'] = data[['math_score','history_score','physics_score',
                          'chemistry_score','biology_score',
                          'english_score','geography_score']].mean(axis=1)

y = data['ExamScore']

# Step 6: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 8: Predictions
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nLinear Regression Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# Step 10: Visualization of predictions
X_test_sorted = np.sort(X_test.values, axis=0)   # Sort for clear line plot
y_pred_sorted = model.predict(X_test_sorted)

plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test_sorted, y_pred_sorted, color="red", linewidth=2, label="Linear Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.legend()
plt.show()

# 🔹 Bonus: Polynomial Regression
poly = PolynomialFeatures(degree=2)  # Try degree=3 too
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

y_poly_pred = poly_model.predict(X_poly_test)

print("\nPolynomial Regression Performance:")
print("R² Score:", r2_score(y_test, y_poly_pred))

# Polynomial Regression Curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_poly_curve = poly_model.predict(poly.transform(X_range))

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X_range, y_poly_curve, color="green", linewidth=2, label="Polynomial Fit")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Polynomial Regression: Study Hours vs Exam Score")
plt.legend()
plt.show()
