import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load Model and Vectorizer
try:
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        raise FileNotFoundError("❌ Model or Vectorizer file not found!")

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
        # Ensure model and vectorizer are loaded
        if model is None or vectorizer is None:
            return jsonify({"error": "Model or vectorizer not loaded"}), 500

        # Parse Request
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request, 'text' key is required"}), 400

        text_input = data["text"]

        # Ensure text_input is a string
        if not isinstance(text_input, str):
            return jsonify({"error": "Invalid input, expected a text string"}), 400

        # Preprocess Text (Lowercase before vectorizing)
        text_input = text_input.strip().lower()

        print(f"🔹 Processed Input: {text_input}")  # Debugging log

        # Vectorize Input (Ensure input is a list of strings)
        transformed_data = vectorizer.transform([text_input])  # List with one string

        print(f"✅ Transformed Data Shape: {transformed_data.shape}")  # Debugging log

        # Convert CSR matrix to Dense Array before passing to model
        transformed_data_dense = transformed_data.toarray()  # Convert to numpy array

        # Make Prediction
        prediction = model.predict(transformed_data_dense)[0]
        result = bool(prediction)

        print(f"✅ Prediction Result: {result}")  # Debugging log

        return jsonify({"spam": result})

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500  # Return full error in response

# Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
