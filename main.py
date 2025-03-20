import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Try loading the model and vectorizer with error handling
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["text"]
        transformed_data = vectorizer.transform([data])  # Convert text using TF-IDF
        prediction = model.predict(transformed_data)[0]
        return jsonify({"spam": bool(prediction)})
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



