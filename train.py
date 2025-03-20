import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin1")

# Preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text

data["message"] = data["message"].astype(str).apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data["message"], data["label"], test_size=0.2)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")
