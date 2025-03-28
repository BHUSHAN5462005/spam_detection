import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample training data (Use your actual dataset)
texts = ["free money now", "win a prize", "hello friend", "how are you"]

# Create and train the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(texts)  # Learn vocabulary from training data

# Save the vectorizer to a file
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Vectorizer saved as vectorizer.pkl")
