import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Define the expected column names
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# 2. Load the CSV without assuming it has a header
data = pd.read_csv("diabetes.csv", header=None, skipinitialspace=True)

# 3. Remove the first row if it contains column headings
first_row = data.iloc[0].astype(str).str.strip().str.lower().tolist()
expected_header = [name.lower() for name in columns]

if first_row == expected_header:
    data = data.iloc[1:].reset_index(drop=True)

# 4. Check that the file has the expected number of columns
if data.shape[1] != len(columns):
    raise ValueError(
        f"Expected {len(columns)} columns in diabetes.csv, "
        f"but found {data.shape[1]}. Check that you are using the correct dataset."
    )

data.columns = columns

# 5. Convert every column to numeric values
for column in columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# Remove rows with missing or invalid target labels
data = data.dropna(subset=["Outcome"]).copy()

# 6. Print basic dataset information
print("Dataset size:", data.shape)
print("\nFirst five rows:")
print(data.head())

print("\nOutcome values:")
print(data["Outcome"].value_counts(dropna=False).sort_index())

# Check that the target contains both classes
unique_outcomes = set(data["Outcome"].unique())

if not unique_outcomes.issubset({0, 1}):
    raise ValueError(
        "The Outcome column should contain only 0 and 1. "
        f"Found these values: {sorted(unique_outcomes)}"
    )

if len(unique_outcomes) < 2:
    raise ValueError(
        "The dataset has only one Outcome class after loading. "
        "Check that diabetes.csv is the correct dataset and that its last "
        "column contains both 0 and 1 labels."
    )

# 7. Replace impossible zero measurements with missing values
# Zero is still allowed for Pregnancies and Outcome.
zero_missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

data[zero_missing_columns] = data[zero_missing_columns].replace(0, float("nan"))

# 8. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"].astype(int)

# 9. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# 10. Create and train the model
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["No diabetes", "Diabetes"],
        zero_division=0
    )
)

# 13. Create and save the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No diabetes", "Diabetes"]
)

display.plot(cmap="Blues")
plt.title("Diabetes Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 14. Save the trained model
joblib.dump(model, "diabetes_prediction_model.pkl")
print("\nModel saved successfully!")