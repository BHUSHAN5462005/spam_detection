<<<<<<< HEAD
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = ["Hello, how are you?", "Win a FREE iPhone now!", "Your account is in danger, click here"]
labels = [0, 1, 1]  # 0 = not spam, 1 = spam

# Train model
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, MultinomialNB())
model.fit(texts, labels)

# Save model as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
=======
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data
texts = ["Hello, how are you?", "Win a FREE iPhone now!", "Your account is in danger, click here"]
labels = [0, 1, 1]  # 0 = not spam, 1 = spam

# Train model
vectorizer = TfidfVectorizer()
model = make_pipeline(vectorizer, MultinomialNB())
model.fit(texts, labels)

# Save model as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")
>>>>>>> 17b3694ad5bf78c70c2f24a12043b6d4bf67532d
