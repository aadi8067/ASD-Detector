from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Expected feature names used during model training
FEATURES = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score",
    "age", "gender", "ethnicity", "jundice", "austim", "relation"
]

# Available models with updated accuracy scores
AVAILABLE_MODELS = {
    "RandomForest": {
        "path": "models/randomforest_model.pkl",
        "accuracy": 95.9
    },
    "LogisticRegression": {
        "path": "models/logisticregression_model.pkl",
        "accuracy": 100.0
    },
    "SVM": {
        "path": "models/svm_model.pkl",
        "accuracy": 95.08
    },
    "KNN": {
        "path": "models/knn_model.pkl",
        "accuracy": 93.44
    }
}

@app.route('/')
def home():
    return render_template('index.html', models=AVAILABLE_MODELS.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # Get selected model from form
        selected_model = form_data.get("model")
        model_info = AVAILABLE_MODELS.get(selected_model)

        if not model_info:
            return jsonify({"error": "Invalid model selected"})

        # Load the selected model
        model = joblib.load(model_info["path"])

        # Collect input features in correct order
        input_data = [
            int(form_data['q1']), int(form_data['q2']), int(form_data['q3']), int(form_data['q4']),
            int(form_data['q5']), int(form_data['q6']), int(form_data['q7']), int(form_data['q8']),
            int(form_data['q9']), int(form_data['q10']),
            float(form_data['age']), int(form_data['gender']), int(form_data['ethnicity']),
            int(form_data['jaundice']), int(form_data['autism']),
            int(form_data['relation'])
        ]

        # Convert input to DataFrame with correct column names
        input_df = pd.DataFrame([input_data], columns=FEATURES)

        # Convert DataFrame to NumPy array before passing to the model
        prediction = model.predict(input_df.to_numpy())[0]

        # Convert result to human-readable format
        result_text = "Autism Detected" if prediction == 1 else "No Autism Detected"

        return jsonify({
            "prediction": result_text,
            "selected_model": selected_model,
            "accuracy": f"{model_info['accuracy']:.2f}%",
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
