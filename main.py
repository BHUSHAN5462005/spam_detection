import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model and vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)  # This might already include vectorizer
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure data is received correctly
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid request, 'text' key is required"}), 400
        
        text_input = data["text"]  # Extract the text
        if not isinstance(text_input, str):
            return jsonify({"error": "Invalid input, expected a string"}), 400
        
        # Directly predict without manually transforming the input
        prediction = model.predict([text_input])[0]
        
        return jsonify({"spam": bool(prediction)})
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
