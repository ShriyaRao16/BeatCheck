"""HEART DISEASE PREDICTION APP USING TKINTER """
#Enter Medical data and predict the risk of heart disease.
#Model trained using Random Forest
#GUI built with Tkinter

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and clean the CSV - FIXED
# The CSV uses tab separator, not comma
df = pd.read_csv("heart.csv", sep="\t")

# Step 2: Clean the data properly
# Remove any leading/trailing spaces from all columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert columns to lowercase for consistency
df.columns = df.columns.str.strip().str.lower()

# Step 3: Print data info for debugging
print("🔍 Dataset Shape:", df.shape)
print("🔍 Columns:", df.columns.tolist())
print("🔍 Target distribution:")
print(df['target'].value_counts())
print("\n🔍 First few rows:")
print(df.head())

# Check for missing values
print("\n🔍 Missing values:")
print(df.isnull().sum())

# Step 4: Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

print("✅ X shape:", X.shape)
print("✅ y shape:", y.shape)
print("✅ Target classes:", y.unique())

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest model with better parameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🔍 Model Accuracy on Test Set: {acc*100:.2f}%")

# Print classification report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Print feature importance
feature_names = X.columns
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n🔍 Feature Importance:")
print(feature_importance_df)

# Test with some sample predictions
print("\n🔍 Testing with sample data:")
sample_high_risk = X_scaled[y == 1][0].reshape(1, -1)
sample_low_risk = X_scaled[y == 0][0].reshape(1, -1)

pred_high = model.predict(sample_high_risk)[0]
prob_high = model.predict_proba(sample_high_risk)[0]
print(f"High risk sample prediction: {pred_high}, probabilities: {prob_high}")

pred_low = model.predict(sample_low_risk)[0]
prob_low = model.predict_proba(sample_low_risk)[0]
print(f"Low risk sample prediction: {pred_low}, probabilities: {prob_low}")

# Feature List with field explanations
features = [
    ("age", "Age (29-77)"),
    ("sex", "Sex (1=Male, 0=Female)"),
    ("cp", "Chest Pain Type (0-3)"),
    ("trestbps", "Resting BP (94-200 mm Hg)"),
    ("chol", "Serum Cholesterol (126-564 mg/dl)"),
    ("fbs", "Fasting Blood Sugar>120? (0/1)"),
    ("restecg", "Resting ECG (0-2)"),
    ("thalach", "Max Heart Rate (71-202 bpm)"),
    ("exang", "Exercise Induced Angina? (0/1)"),
    ("oldpeak", "ST Depression (0.0-6.2)"),
    ("slope", "Slope of ST (0-2)"),
    ("ca", "Number of Major Vessels (0-3) by Fluoroscopy "),
    ("thal", "Thalassemia (1=Normal, 2=Fixed, 3=Reversible)")
]

# GUI - Creating the Tkinter Window
app = tk.Tk()
app.title("Heart Disease Prediction")
app.geometry("450x700")
app.resizable(False, False)

# Title label
title_label = tk.Label(app, text="BeatCheck", font=("Segoe UI", 20, "bold"), fg="darkblue")
title_label.pack(pady=15)

# Subtitle
subtitle_label = tk.Label(app, text="Heart Disease Risk Prediction", font=("Segoe UI", 12), fg="gray")
subtitle_label.pack(pady=(0, 10))

# Main frame
main_frame = tk.Frame(app)
main_frame.pack(pady=10, padx=20, fill="both", expand=True)

# Create scrollable frame
canvas = tk.Canvas(main_frame)
scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Form frame
form = tk.Frame(scrollable_frame)
form.pack(pady=10, padx=10)

entries = {}

# Create input fields
for field_key, field_label in features:
    row = tk.Frame(form)
    row.pack(pady=5, fill="x")

    label = tk.Label(row, text=field_label, width=35, anchor="w", font=("Arial", 10))
    label.pack(side="left")

    entry = tk.Entry(row, width=15, font=("Arial", 10))
    entry.pack(side="right", padx=(10, 0))

    entries[field_key] = entry

# Pack canvas and scrollbar
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# Predict Function - FIXED
def predict():
    try:
        # Get user input
        user_input = []
        for field_key, _ in features:
            value = entries[field_key].get().strip()
            if value == "":
                messagebox.showerror("Input Error", f"Please enter a value for {field_key}")
                return
            user_input.append(float(value))

        # Convert to numpy array and reshape
        user_input_array = np.array(user_input).reshape(1, -1)

        # Scale the input using the same scaler
        user_input_scaled = scaler.transform(user_input_array)

        # Make prediction
        prediction = model.predict(user_input_scaled)[0]
        probabilities = model.predict_proba(user_input_scaled)[0]

        # Get confidence for the predicted class
        confidence = probabilities[prediction]

        # Debug print
        print(f"\n🔍 User Input: {user_input}")
        print(f"🔍 Scaled Input: {user_input_scaled}")
        print(f"🔍 Prediction: {prediction}")
        print(f"🔍 Probabilities: {probabilities}")
        print(f"🔍 Confidence: {confidence*100:.2f}%")

        # Show result
        if prediction == 1:
            messagebox.showwarning(
                "Prediction Result",
                f"✅ LOW RISK of Heart Disease!\n\n"
                f"Confidence: {confidence*100:.2f}%\n"
                f"Risk Probability: {probabilities[1]*100:.2f}%\n\n"
                f"Please consult a cardiologist for proper evaluation."
            )
        else:
            messagebox.showinfo(
                "Prediction Result",
                f"⚠️HIGH RISK of Heart Disease\n\n"
                f"Confidence: {confidence*100:.2f}%\n"
                f"Risk Probability: {probabilities[1]*100:.2f}%\n\n"
                f"Continue maintaining a healthy lifestyle!"
            )

    except ValueError as e:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Clear Function
def clear_fields():
    for entry in entries.values():
        entry.delete(0, tk.END)

# Sample data function for testing
def load_sample_high_risk():
    # Sample high-risk patient data
    sample_data = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]
    for i, (field_key, _) in enumerate(features):
        entries[field_key].delete(0, tk.END)
        entries[field_key].insert(0, str(sample_data[i]))

def load_sample_low_risk():
    # Sample low-risk patient data
    sample_data = [29, 0, 0, 120, 180, 0, 0, 170, 0, 0, 2, 0, 2]
    for i, (field_key, _) in enumerate(features):
        entries[field_key].delete(0, tk.END)
        entries[field_key].insert(0, str(sample_data[i]))

# Button frame
button_frame = tk.Frame(app)
button_frame.pack(pady=20)

# Buttons
predict_btn = tk.Button(button_frame, text="Predict Risk", command=predict,
                       bg="lightblue", font=("Arial", 12, "bold"),
                       width=12, height=2)
predict_btn.pack(side="left", padx=5)

clear_btn = tk.Button(button_frame, text="Clear All", command=clear_fields,
                     bg="lightgray", font=("Arial", 12),
                     width=12, height=2)
clear_btn.pack(side="left", padx=5)

# Sample data buttons
sample_frame = tk.Frame(app)
sample_frame.pack(pady=(0, 20))

high_risk_btn = tk.Button(sample_frame, text="Load High Risk Sample",
                         command=load_sample_high_risk,
                         bg="lightcoral", font=("Arial", 10),
                         width=20)
high_risk_btn.pack(side="left", padx=5)

low_risk_btn = tk.Button(sample_frame, text="Load Low Risk Sample",
                        command=load_sample_low_risk,
                        bg="lightgreen", font=("Arial", 10),
                        width=20)
low_risk_btn.pack(side="left", padx=5)

# Run the app
if __name__ == "__main__":
    app.mainloop()