import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("spam_ham_dataset.csv", encoding="latin1")

# Preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

data["text"] = data["text"].astype(str).apply(clean_text)

# Map labels correctly
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Train-test split (Ensure balanced training)
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"])

# Convert text to TF-IDF (Increase features slightly but avoid overfitting)
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words='english')  # Added stop words removal
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model (Adjust alpha to prevent overfitting)
model = MultinomialNB(alpha=0.1)  # Increased alpha slightly to avoid false positives
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")

# ✅ Sanity Check: Test model with both spam and ham
test_texts = ["Congratulations! You won a free iPhone 15", "hi", "Hello, how are you?", "Click here to win $$$!"]
transformed_texts = vectorizer.transform(test_texts)
predictions = model.predict(transformed_texts)

for text, pred in zip(test_texts, predictions):
    print(f"Test: '{text}' → Prediction: {'spam' if pred == 1 else 'ham'}")  # ✅ "hi" should now be "ham"
