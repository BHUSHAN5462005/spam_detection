import pickle
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Spam Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("text", "")  # Handle missing 'text' key safely
    if not data:
        return jsonify({"error": "No text provided"}), 400  # Return error if no text
    
    prediction = model.predict([data])[0]
    return jsonify({"spam": bool(prediction)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)


