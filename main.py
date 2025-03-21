from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# ✅ Load the trained model and vectorizer
MODEL_PATH = "spam_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError("Model or vectorizer file not found. Train the model first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# ✅ Health check route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Spam Detection API is running!"})

# ✅ Spam prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"]

        # ✅ Transform input text and make prediction
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]

        return jsonify({"prediction": "spam" if prediction == 1 else "ham"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Start the Flask app on the correct port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port)
