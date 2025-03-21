import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("spam_ham_dataset.csv", encoding="latin1")

# Convert labels to numerical format if necessary
if data["label"].dtype == "object":
    data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep words and numbers
    return text

data["text"] = data["text"].astype(str).apply(clean_text)

# Train-test split (stratified to keep spam/ham balance)
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=42
)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")

