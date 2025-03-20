import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Model & Vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")

@app.route("/", methods=["GET"])
def home():
    return "Spam Detection API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate Input
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request, 'text' key is required"}), 400
        
        text_input = data["text"].strip()
        if not isinstance(text_input, str) or len(text_input) == 0:
            return jsonify({"error": "Invalid input, expected a non-empty string"}), 400

        # Convert text using TF-IDF
        transformed_data = vectorizer.transform([text_input])

        # Get Prediction & Probability
        prediction = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0][1]  # Spam probability

        return jsonify({
            "text": text_input,
            "spam": bool(prediction),
            "confidence": round(probability, 3)
        })
    
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
