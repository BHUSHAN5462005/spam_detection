from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and vectorizer (🔥 Make sure names match train.py)
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email_text = data.get('email', '')

    # Debugging: Print received email
    print("\nReceived Email for Prediction:\n", email_text)

    # Transform email text using TF-IDF vectorizer
    email_tfidf = vectorizer.transform([email_text])

    # Debugging: Print top TF-IDF features
    feature_names = vectorizer.get_feature_names_out()
    sorted_indices = email_tfidf.toarray()[0].argsort()[::-1]

    print("\nTop 10 TF-IDF Features:")
    for index in sorted_indices[:10]:
        print(feature_names[index], ":", email_tfidf[0, index])

    # Predict spam or ham
    prediction = model.predict(email_tfidf)[0]
    result = "spam" if prediction == 1 else "ham"

    print("\nPrediction:", result)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
