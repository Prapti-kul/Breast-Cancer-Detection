import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

st.set_page_config(
    page_title="Prapti's Breast Cancer Detection",
    page_icon="🎗️",
    layout="wide"
)

st.title("Prapti's Breast Cancer Detection Dashboard")
st.write("Logistic Regression vs Support Vector Machine (SVM)")

#---------------------------------------
# Step 1 : Check required files
#---------------------------------------

required_files = [
    "logistic_regression_model.pkl",
    "svm_model.pkl",
    "breast_cancer_model_results.csv",
    "logistic_regression_confusion_matrix.png",
    "svm_confusion_matrix.png",
    "breast_cancer_model_comparison.png"
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]

if missing_files:
    st.warning("Please run BreastCancer_Detection.py first.")
    st.write("Missing files:", missing_files)
    st.stop()

#---------------------------------------
# Step 2 : Load saved models and results
#---------------------------------------

logistic_model = joblib.load("logistic_regression_model.pkl")
svm_model = joblib.load("svm_model.pkl")

results = pd.read_csv("breast_cancer_model_results.csv")

#---------------------------------------
# Step 3 : Display model performance
#---------------------------------------

st.subheader("1. Model Performance")

col1, col2 = st.columns(2)

logistic_results = results[
    results["Model"] == "Logistic Regression"
]

svm_results = results[
    results["Model"] == "Support Vector Machine"
]

if not logistic_results.empty and not svm_results.empty:

    col1.metric(
        "Logistic Regression Accuracy",
        f"{logistic_results['Accuracy'].iloc[0] * 100:.2f}%"
    )

    col2.metric(
        "SVM Accuracy",
        f"{svm_results['Accuracy'].iloc[0] * 100:.2f}%"
    )

st.dataframe(results, use_container_width=True)

#---------------------------------------
# Step 4 : Compare model metrics
#---------------------------------------

st.subheader("2. Model Comparison")

results_melted = results.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
    var_name="Metric",
    value_name="Score"
)

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=results_melted,
    x="Metric",
    y="Score",
    hue="Model",
    ax=ax
)

ax.set_title("Logistic Regression vs SVM")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)

st.pyplot(fig)
plt.close(fig)

#---------------------------------------
# Step 5 : Display confusion matrix
#---------------------------------------

st.subheader("3. Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.write("Logistic Regression")
    st.image("logistic_regression_confusion_matrix.png")

with col2:
    st.write("Support Vector Machine")
    st.image("svm_confusion_matrix.png")

#---------------------------------------
# Step 6 : Predict using saved models
#---------------------------------------

st.subheader("4. Predict Using a Saved Model")

st.write(
    "Enter the feature values from a record in your dataset. "
    "The number and order of features must match the training data."
)

selected_model = st.selectbox(
    "Select Model",
    ["Logistic Regression", "Support Vector Machine"]
)

st.info(
    "Your model uses the feature columns from breast_cancer.csv. "
    "For reliable predictions, use values in the same format and order "
    "as the original dataset."
)

#---------------------------------------
# Step 7 : Download model results
#---------------------------------------

st.subheader("5. Download Results")

st.download_button(
    "Download Model Comparison CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="breast_cancer_model_results.csv",
    mime="text/csv"
)

st.success("Dashboard loaded successfully!")

st.caption(
    "Educational machine learning demonstration only. "
    "This application is not a clinically validated diagnostic tool."
)
#---------------------------------------
# Step 7 : Test with new data
#---------------------------------------

st.subheader("6. Test with New Data")

st.write("Enter feature values to test a new sample.")

# Load feature names from the training dataset
df = pd.read_csv("breast_cancer.csv")
feature_names = df.drop("target", axis=1).columns.tolist()

with st.form("new_data_form"):

    st.write("Enter the values for all features:")

    input_values = {}

    for feature in feature_names:
        input_values[feature] = st.number_input(
            feature,
            value=float(df[feature].median()),
            format="%.4f"
        )

    test_button = st.form_submit_button("Predict")

if test_button:

    new_data = pd.DataFrame(
        [input_values],
        columns=feature_names
    )

    if selected_model == "Logistic Regression":
        prediction = logistic_model.predict(new_data)[0]
        model_used = logistic_model
    else:
        prediction = svm_model.predict(new_data)[0]
        model_used = svm_model

    st.subheader("Prediction Result")

    st.write("Selected Model:", selected_model)

    st.write("Predicted class:", prediction)

    st.info(
        "Check your dataset's target encoding to identify "
        "which numeric label represents benign or malignant."
    )

    if hasattr(model_used, "predict_proba"):
        probabilities = model_used.predict_proba(new_data)[0]

        probability_df = pd.DataFrame({
            "Class": model_used.classes_,
            "Estimated Probability": probabilities
        })

        st.write("Estimated class probabilities:")
        st.dataframe(probability_df, use_container_width=True)