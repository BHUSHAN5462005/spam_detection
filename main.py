from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow all origins (for testing)

# ✅ Load your trained model & vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return jsonify({"message": "Spam Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # ✅ Transform input text and make prediction
        transformed_text = vectorizer.transform([text])
        prediction = model.predict(transformed_text)[0]

        return jsonify({"prediction": "spam" if prediction == 1 else "ham"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
