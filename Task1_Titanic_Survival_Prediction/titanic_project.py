# ============================================
# Titanic Survival Prediction
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2️⃣ Create outputs folder if not exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 3️⃣ Load Dataset
df = pd.read_csv("data/Titanic-Dataset.csv")

print("First 5 rows:\n", df.head())

# 4️⃣ Data Cleaning

# Fill missing numeric values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Fill missing categorical values
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop Cabin (too many missing values)
if 'Cabin' in df.columns:
    df = df.drop('Cabin', axis=1)

# Drop unnecessary columns safely
columns_to_drop = ['PassengerId', 'Name', 'Ticket']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop(col, axis=1)

# 5️⃣ Convert Categorical Data

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# 6️⃣ Final Check for Missing Values
print("\nRemaining Missing Values:\n", df.isnull().sum())

# 7️⃣ Visualization

plt.figure()
sns.countplot(x='Survived', data=df)
plt.title("Survival Distribution")
plt.savefig("outputs/survival_distribution.png")
plt.close()

# 8️⃣ Define Features and Target
X = df.drop('Survived', axis=1)
y = df['Survived']

# 9️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔟 Train Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 1️⃣1️⃣ Predictions
y_pred = model.predict(X_test)

# 1️⃣2️⃣ Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 1️⃣3️⃣ Save Model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")
