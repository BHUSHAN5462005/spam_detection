import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    prediction = model.predict([data])[0]
    return jsonify({"spam": bool(prediction)})

if __name__ == "__main__":
    app.run(debug=True)

