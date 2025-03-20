import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# 1️⃣ Load dataset (replace with your dataset)
data = pd.read_csv("spam_ham_dataset.csv")  # Ensure you have a dataset
X = data["text"]  # Replace with actual column name
y = data["label"]    # Replace with actual column name (0 = Ham, 1 = Spam)

# 2️⃣ Convert text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# 3️⃣ Train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_transformed, y)

# 4️⃣ Save the model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
