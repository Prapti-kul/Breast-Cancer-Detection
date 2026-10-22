import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

#---------------------------------------
# Step 1 : Load the dataset
#---------------------------------------

df = pd.read_csv("breast_cancer.csv")

print("Shape of dataset:", df.shape)
print("First few records:")
print(df.head())

#---------------------------------------
# Step 2 : Separate features and labels
#---------------------------------------

X = df.drop("target", axis=1)
Y = df["target"]

print("X shape:", X.shape)
print("Y shape:", Y.shape)

#---------------------------------------
# Step 3 : Split dataset for training and testing
#---------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

#---------------------------------------
# Step 4 : Create individual models
#---------------------------------------

model_log = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000))
])

model_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(kernel="rbf"))
])

models = {
    "Logistic Regression": model_log,
    "Support Vector Machine": model_svm
}

results = []

#---------------------------------------
# Step 5 : Train and evaluate models
#---------------------------------------

for name, model in models.items():

    print("\n---------------------------------------")
    print(name)
    print("---------------------------------------")

    # Train the model
    model.fit(X_train, Y_train)

    # Test the model
    Y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, zero_division=0)
    recall = recall_score(Y_test, Y_pred, zero_division=0)
    f1 = f1_score(Y_test, Y_pred, zero_division=0)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    print("Confusion Matrix:")
    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)

    # Save model
    if name == "Logistic Regression":
        joblib.dump(model, "logistic_regression_model.pkl")
    else:
        joblib.dump(model, "svm_model.pkl")

    # Save confusion matrix image
    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    filename = (
        "logistic_regression_confusion_matrix.png"
        if name == "Logistic Regression"
        else "svm_confusion_matrix.png"
    )

    plt.savefig(filename)
    plt.close()

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

#---------------------------------------
# Step 6 : Save comparison results
#---------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "breast_cancer_model_results.csv",
    index=False
)

print("\nModel Comparison:")
print(results_df)

#---------------------------------------
# Step 7 : Plot model comparison
#---------------------------------------

results_melted = results_df.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_melted,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title("Logistic Regression vs SVM")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("breast_cancer_model_comparison.png")
plt.show()

print("\nModels and results saved successfully!")
