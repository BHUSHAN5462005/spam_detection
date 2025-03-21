from flask import Flask, request, jsonify
import joblib
import os
app = Flask(__name__)  # ✅ Ensure Flask app is initialized

# ✅ Load the trained model and vectorizer
try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    print("❌ Model or vectorizer file not found. Ensure 'spam_model.pkl' and 'vectorizer.pkl' exist.")

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
    port = int(os.environ.get("PORT", 5000))  # Get port from Render
    app.run(host="0.0.0.0", port=port)  # Bind to all IPs
