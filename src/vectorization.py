import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
df = pd.read_csv("data/judgments.csv")

# load cleaned text (recreate if needed)
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['judgment_text'].apply(clean_text)

# TF-IDF vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['clean_text'])

# output shape
print("Shape of TF-IDF matrix:", X.shape)

# show feature names
print("Sample features:", vectorizer.get_feature_names_out()[:10])