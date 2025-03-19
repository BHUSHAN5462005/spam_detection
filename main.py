import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    transformed_data = vectorizer.transform([data])  # Convert text using TF-IDF
    prediction = model.predict(transformed_data)[0]
    return jsonify({"spam": bool(prediction)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)



