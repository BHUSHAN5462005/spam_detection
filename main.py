import pickle
from flask import Flask, request, jsonify
import traceback  # Helps capture detailed error logs

app = Flask(__name__)

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)  # Ensure TF-IDF Vectorizer is loaded
except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    traceback.print_exc()

@app.route("/", methods=["GET"])
def home():
    return "Spam Detection API is running!", 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure JSON data is received
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request, 'text' key is required"}), 400
        
        text_input = data["text"]
        if not isinstance(text_input, str) or not text_input.strip():
            return jsonify({"error": "Invalid input, expected a non-empty string"}), 400

        # Transform text using TF-IDF vectorizer
        transformed_data = vectorizer.transform([text_input])
        transformed_data = transformed_data.toarray()  # Convert to dense array

        # Predict
        prediction = model.predict(transformed_data)[0]

        return jsonify({"spam": bool(prediction)})

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()  # Print full error trace in logs
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
