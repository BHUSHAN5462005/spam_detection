import pandas as pd
import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
data = pd.read_csv("spam_ham_dataset.csv", encoding="latin1")

# Verify labels
print("Unique labels in dataset:", data["label"].unique())

# Convert labels if necessary
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Ensure no missing values
data = data.dropna()

# Text Preprocessing Function
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization
    return text

data["text"] = data["text"].astype(str).apply(clean_text)

# Print a few processed texts
print("Sample processed texts:", data["text"].head())

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, stratify=data["label"], random_state=42)

# Convert text to TF-IDF
vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
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

# Testing with sample inputs
test_texts = [
    "Congratulations! You won a free iPhone 15",
    "hi",
    "Hello, how are you?",
    "Click here to win $$$!",
    "Your order has been shipped, tracking number is 12345",
    "Claim your free Bitcoin now!"
]

for text in test_texts:
    transformed_text = vectorizer.transform([clean_text(text)])
    prediction = model.predict(transformed_text)[0]
    print(f"Test: '{text}' → Prediction: {'spam' if prediction == 1 else 'ham'}")
