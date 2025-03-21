import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 🔹 Manually defined stopwords (since NLTK can't download)
custom_stopwords = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at",
    "by", "for", "with", "about", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

# 🔹 Load dataset and fix column names
data = pd.read_csv("spam.csv", encoding="latin1")
data = data.rename(columns={"v1": "label", "v2": "text"})  # Rename relevant columns
data = data[["label", "text"]]  # Keep only the required columns

# 🔹 Convert labels to numerical (0=ham, 1=spam)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# 🔹 Preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    words = text.split()
    
    # Keep important spam-related words
    important_words = {"free", "won", "congratulations", "prize", "click", "offer", "buy", "sale"}
    
    words = [word for word in words if word not in custom_stopwords or word in important_words]
    return " ".join(words)

# 🔹 Apply preprocessing
data["text"] = data["text"].astype(str).apply(clean_text)

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# 🔹 Improved TF-IDF settings
vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=1, max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 🔹 Train the Multinomial Naïve Bayes model
model = MultinomialNB(alpha=0.1)  # Lower alpha improves spam recall
model.fit(X_train_vec, y_train)

# 🔹 Model evaluation
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 🔹 Save the model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully!")

# 🔹 Quick test predictions
test_texts = [
    "Congratulations! You won a free iPhone 15",
    "hi",
    "Hello, how are you?",
    "Click here to win $$$!",
    "Your order has been shipped, tracking number is 12345",
    "Claim your free Bitcoin now!"
]

test_texts_cleaned = [clean_text(text) for text in test_texts]
test_vec = vectorizer.transform(test_texts_cleaned)
test_predictions = model.predict(test_vec)

for text, pred in zip(test_texts, test_predictions):
    print(f"Test: '{text}' → Prediction: {'spam' if pred == 1 else 'ham'}")
