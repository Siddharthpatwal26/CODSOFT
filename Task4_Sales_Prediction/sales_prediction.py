# ============================================
# Sales Prediction Using Python
# CODSOFT Data Science Internship
# ============================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2️⃣ Create outputs folder if not exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 3️⃣ Load Dataset
df = pd.read_csv("data/advertising.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# 4️⃣ Check Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# 5️⃣ Data Visualization

# Sales distribution
plt.figure()
sns.histplot(df['Sales'], bins=20)
plt.title("Sales Distribution")
plt.savefig("outputs/sales_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# 6️⃣ Define Features and Target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8️⃣ Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = model.predict(X_test)

# 🔟 Evaluation

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# 1️⃣1️⃣ Regression Plot (Actual vs Predicted)
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.savefig("outputs/regression_plot.png")
plt.close()

# 1️⃣2️⃣ Save Model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")
