import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Model and Vectorizer
try:
    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("❌ Model file 'model.pkl' not found!")
    if not os.path.exists("vectorizer.pkl"):
        raise FileNotFoundError("❌ Vectorizer file 'vectorizer.pkl' not found!")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    print("✅ Model and Vectorizer loaded successfully!")

except Exception as e:
    print(f"❌ Error loading model/vectorizer: {e}")
    vectorizer, model = None, None  # Prevent execution if loading fails

# Health Check Route
@app.route("/", methods=["GET"])
def home():
    return "Spam Detection API is running!", 200

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model and vectorizer are loaded
        if model is None or vectorizer is None:
            print("❌ Model or Vectorizer is not loaded properly!")
            return jsonify({"error": "Model or vectorizer not loaded"}), 500

        # Parse Request
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request, 'text' key is required"}), 400

        text_input = data["text"]

        # Validate Input
        if not isinstance(text_input, str):
            return jsonify({"error": "Invalid input, expected a text string"}), 400

        text_input = text_input.strip().lower()  # Ensure it's clean text

        print(f"🔹 Processed Input: {text_input}")  # Debugging log

        # Vectorize Input
        transformed_data = vectorizer.transform([text_input])

        # Ensure it's in the correct format
        transformed_data_dense = transformed_data.toarray()

        print(f"✅ Dense Data Shape: {transformed_data_dense.shape}")  # Debugging log

        # Make Prediction
        prediction = model.predict(transformed_data_dense)[0]
        print(f"✅ Prediction Result: {prediction}")  # Debugging log

        return jsonify({"spam": bool(prediction)})

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()  # Print full error trace in logs
        return jsonify({"error": "Internal Server Error"}), 500

# Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
