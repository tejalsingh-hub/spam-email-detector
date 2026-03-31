import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("../dataset/spam.csv", encoding='latin-1')

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels (ham = 0, spam = 1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text into numbers
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Check accuracy
accuracy = model.score(X_test_vec, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
pickle.dump(model, open("../model/model.pkl", "wb"))
pickle.dump(vectorizer, open("../model/vectorizer.pkl", "wb"))

# Prediction loop
while True:
    msg = input("\nEnter a message (or type 'exit'): ")
    if msg.lower() == 'exit':
        break
    
    vec = vectorizer.transform([msg])
    result = model.predict(vec)[0]
    
    if result == 1:
        print("Spam ")
    else:
        print("Not Spam ")