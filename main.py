import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load dataset
data = pd.read_csv("spam_ham_dataset.csv")

# Use the correct column names
X = data["text"]  # Change "text" to the actual column name
y = data["label"]  # Change "label" to the actual column name (0 = Ham, 1 = Spam)

# Convert text data into numerical form using TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Train the Naïve Bayes model
model = MultinomialNB()
model.fit(X_transformed, y)

# Save the model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
