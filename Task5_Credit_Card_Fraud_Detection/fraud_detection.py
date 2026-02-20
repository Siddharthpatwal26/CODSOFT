# ============================================
# Credit Card Fraud Detection
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score
)
from sklearn.ensemble import RandomForestClassifier

# 1️⃣ Create outputs folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# 2️⃣ Load dataset
df = pd.read_csv("data/creditcard.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Shape:", df.shape)

# 3️⃣ Check missing values
print("\nMissing Values:\n", df.isnull().sum())

# 4️⃣ Check class distribution
print("\nClass Distribution:\n", df['Class'].value_counts())

# 5️⃣ Scale Amount column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

# 6️⃣ Define features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 7️⃣ Train-test split (important: stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8️⃣ Train Random Forest (better for imbalance)
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1   # uses all CPU cores
)

print("\nTraining model...")
model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = model.predict(X_test)

# 🔟 Evaluation
accuracy = accuracy_score(y_test, y_pred)
fraud_recall = recall_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Fraud Recall (Class 1):", fraud_recall)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 1️⃣1️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

# 1️⃣2️⃣ Save Model
joblib.dump(model, "model.pkl")

print("\nModel saved successfully as model.pkl")
