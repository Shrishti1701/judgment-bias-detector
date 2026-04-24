import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

# download stopwords (only first time)
nltk.download('stopwords')

# load dataset
df = pd.read_csv("data/judgments.csv")

# stopwords
stop_words = set(stopwords.words('english'))

# cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# apply cleaning
df['clean_text'] = df['judgment_text'].apply(clean_text)

# show output
print(df[['judgment_text', 'clean_text']].head())