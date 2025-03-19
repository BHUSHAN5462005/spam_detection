from flask import Flask, request, jsonify
import pickle
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "Spam Detection Model API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    transformed_text = vectorizer.transform([data])
    prediction = model.predict(transformed_text)[0]
    return jsonify({'prediction': 'Spam' if prediction == 1 else 'Not Spam'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

