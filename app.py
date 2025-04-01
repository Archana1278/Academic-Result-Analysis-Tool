import matplotlib
matplotlib.use('Agg')  # Prevent GUI-related errors
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Load dataset
file_path = "improved_dataset.csv"
df = pd.read_csv(file_path)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

# Feature Engineering: Add new columns
df['sgpa_change_1_to_2'] = df['2nd sem sgpa'] - df['1st sem sgpa']
df['sgpa_change_2_to_3'] = df['3rd sem sgpa'] - df['2nd sem sgpa']
df['attendance_ratio'] = df['attendance'] / 100
df['assignment_internal_interaction'] = df['assignment_score'] * df['internal_exam']
df['attendance_squared'] = df['attendance'] ** 2

# Ensure dataset has necessary features
required_columns = [
    "1st sem sgpa", "2nd sem sgpa", "attendance", "assignment_score", "internal_exam",
    "sgpa_change_1_to_2", "sgpa_change_2_to_3", "attendance_ratio",
    "assignment_internal_interaction", "attendance_squared"
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Prepare dataset
X = df[required_columns]
y_3 = df["2nd sem sgpa"] * 1.05  # Assume slight improvement
y_4 = df["2nd sem sgpa"] * 1.03

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y3_train, y3_test = train_test_split(X_scaled, y_3, test_size=0.2, random_state=42)
X_train, X_test, y4_train, y4_test = train_test_split(X_scaled, y_4, test_size=0.2, random_state=42)

# Train model with optimized parameters
model_3 = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=10, objective='reg:squarederror', random_state=0)
model_3.fit(X_train, y3_train)

model_4 = XGBRegressor(n_estimators=500, learning_rate=0.1, max_depth=10, objective='reg:squarederror', random_state=0)
model_4.fit(X_train, y4_train)

# Compute R² Score
accuracy_3 = r2_score(y3_test, model_3.predict(X_test))
accuracy_4 = r2_score(y4_test, model_4.predict(X_test))
print(f"Prediction Accuracy for 3rd Sem SGPA: {accuracy_3:.2f}")
print(f"Prediction Accuracy for 4th Sem SGPA: {accuracy_4:.2f}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get("sgpa_values", [])
        if len(input_data) < 1 or len(input_data) > 3:
            return jsonify({"error": "Please enter between 1 and 3 SGPA values"}), 400

        predicted_sgpa = []
        total_semesters = 4

        while len(input_data) + len(predicted_sgpa) < total_semesters:
            model = model_3 if len(input_data) + len(predicted_sgpa) < 3 else model_4

            # Prepare features for prediction
            features = input_data + predicted_sgpa
            while len(features) < 2:
                features.append(np.mean(df[["1st sem sgpa", "2nd sem sgpa"]].values))

            # Add default values for derived features
            attendance = 85
            assignment_score = 90
            internal_exam = 95

            # Calculate derived features
            sgpa_change_1_to_2 = features[1] - features[0] if len(features) >= 2 else 0
            sgpa_change_2_to_3 = features[2] - features[1] if len(features) >= 3 else 0
            attendance_ratio = attendance / 100
            assignment_internal_interaction = assignment_score * internal_exam
            attendance_squared = attendance ** 2

            # Combine all features
            input_features = [
                features[0], features[1],
                attendance,
                assignment_score,
                internal_exam,
                sgpa_change_1_to_2,
                sgpa_change_2_to_3,
                attendance_ratio,
                assignment_internal_interaction,
                attendance_squared
            ]

            # Scale input features
            input_features = pd.DataFrame([input_features], columns=required_columns)
            input_features = scaler.transform(input_features)

            # Predict SGPA
            predicted_value = round(float(model.predict(input_features)[0]), 2)
            predicted_sgpa.append(max(predicted_value, features[0] * 0.95))  # Prevent drastic drop

        # Combine actual and predicted SGPA
        all_sgpa = input_data + predicted_sgpa
        predicted_cgpa = round(sum(all_sgpa) / len(all_sgpa), 2)

        # Generate graph
        sem_labels = [f"{i+1} Sem" for i in range(len(all_sgpa))]
        plt.figure(figsize=(10, 6))
        plt.plot(sem_labels, all_sgpa, marker='o', linestyle='-', color='black', label="SGPA Trend")
        plt.plot(sem_labels[:len(input_data)], input_data, marker='o', linestyle='-', color='orange', label="Actual SGPA")
        plt.plot(sem_labels[len(input_data):], predicted_sgpa, marker='o', linestyle='-', color='blue', label="Predicted SGPA")
        for i, txt in enumerate(all_sgpa):
            plt.text(i, txt + 0.1, str(txt), fontsize=12, ha='center')
        plt.xlabel("Semester")
        plt.ylabel("SGPA")
        plt.title("SGPA & CGPA Prediction")
        plt.legend()
        plt.grid(True)

        img_path = "static/prediction_graph.png"
        plt.savefig(img_path, bbox_inches='tight')
        plt.close()

        return jsonify({
            "predicted_sgpa": all_sgpa,
            "predicted_cgpa": predicted_cgpa,
            "graph_url": img_path
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/graph')
def get_graph():
    return send_file("static/prediction_graph.png", mimetype='image/png')


if __name__ == '__main__':
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)