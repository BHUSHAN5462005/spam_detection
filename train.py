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

# Check label encoding
print("Label Distribution:\n", data["label"].value_counts())

# Ensure spam is labeled as 1 and ham as 0
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Train-test split (fixed random_state)
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"])

# Convert text to TF-IDF (increase max_features)
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))  # Increase feature size and use bigrams
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with better alpha
model = MultinomialNB(alpha=0.05)  # Lower alpha for better spam classification
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

# ✅ Sanity Check: Test model with a sample spam message
test_text = ["Congratulations! You won a free iPhone 15"]
transformed_text = vectorizer.transform(test_text)
prediction = model.predict(transformed_text)[0]
print("Test Prediction:", "spam" if prediction == 1 else "ham")  # ✅ Should print "spam"
