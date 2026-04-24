import streamlit as st
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------------------
# Setup
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/judgments.csv")
df['clean_text'] = df['judgment_text'].apply(clean_text)

# -------------------------------
# Model Training
# -------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['outcome']

model = LogisticRegression()
model.fit(X, y)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Judgment Bias Detector", layout="centered")

st.title("⚖️ AI-Powered Judgment Bias Detector")
st.markdown("Analyze court judgments using NLP & Machine Learning to detect patterns and bias.")

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔍 Predict Case Outcome")

user_input = st.text_area("Enter judgment text:")

if st.button("Predict Outcome"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        st.success(f"📌 Predicted Outcome: {prediction}")
    else:
        st.warning("Please enter some text")

# -------------------------------
# Bias Analysis
# -------------------------------
st.subheader("⚖️ Bias Analysis")

# Gender Bias
st.markdown("### Gender vs Outcome")
gender_bias = pd.crosstab(df['gender'], df['outcome'], normalize='index')
st.dataframe(gender_bias)

# Plot Gender Bias
fig1, ax1 = plt.subplots()
gender_bias.plot(kind='bar', ax=ax1)
ax1.set_title("Gender Bias Distribution")
st.pyplot(fig1)

# Region Bias
st.markdown("### Region vs Outcome")
region_bias = pd.crosstab(df['region'], df['outcome'], normalize='index')
st.dataframe(region_bias)

# Plot Region Bias
fig2, ax2 = plt.subplots()
region_bias.plot(kind='bar', ax=ax2)
ax2.set_title("Region Bias Distribution")
st.pyplot(fig2)

# -------------------------------
# Insights
# -------------------------------
st.subheader("📊 Key Insights")

try:
    if 'Male' in gender_bias.index and 'Female' in gender_bias.index:
        male_guilty = gender_bias.loc['Male'].get('Guilty', 0)
        female_guilty = gender_bias.loc['Female'].get('Guilty', 0)

        if male_guilty > female_guilty:
            st.warning("⚠️ Higher 'Guilty' rate observed for Male → potential bias")
        elif female_guilty > male_guilty:
            st.warning("⚠️ Higher 'Guilty' rate observed for Female → potential bias")
        else:
            st.info("No major gender bias observed in 'Guilty' outcomes")
except:
    st.info("Not enough data for bias insight")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 Built using NLP + Machine Learning | Streamlit App")