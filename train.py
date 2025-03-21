import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# 1️⃣ Load dataset
data = pd.read_csv("spam_ham_dataset.csv", encoding="latin1")

# 2️⃣ Check and Fix Labels
print("Label distribution before:", data["label"].value_counts())
if data["label"].dtype == object:
    data["label"] = data["label"].map({"ham": 0, "spam": 1})  # Ensure labels are 0 (ham) and 1 (spam)
print("Label distribution after mapping:", data["label"].value_counts())

# 3️⃣ Preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text

data["text"] = data["text"].astype(str).apply(clean_text)

# 4️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# 5️⃣ Convert text to TF-IDF with better settings
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6️⃣ Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7️⃣ Save Model and Vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")

# 8️⃣ Test a sample prediction
test_text = ["Congratulations! You won a free iPhone 15"]
transformed_text = vectorizer.transform(test_text)
prediction = model.predict(transformed_text)[0]
print("Test Prediction:", "spam" if prediction == 1 else "ham")
