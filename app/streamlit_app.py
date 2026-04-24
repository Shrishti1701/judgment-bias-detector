import streamlit as st
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Judgment Bias Detector",
    layout="wide",
    page_icon="⚖️"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00BFFF;
    }
    .stButton>button {
        background-color: #00BFFF;
        color: white;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("About")
st.sidebar.info("""
This app uses NLP and Machine Learning to analyze court judgments,
predict outcomes, and detect potential bias across gender and regions.
""")

st.sidebar.write(f"📊 Model Accuracy: {accuracy:.2f}")

st.sidebar.subheader("Example Input")
st.sidebar.write("The accused was found guilty due to strong evidence.")

# -------------------------------
# Main UI
# -------------------------------
st.title("⚖️ AI-Powered Judgment Bias Detector")
st.markdown("Analyze court judgments using NLP & Machine Learning.")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Enter Judgment Text")
    user_input = st.text_area("")

with col2:
    st.subheader("📌 Prediction Result")
    if st.button("Predict Outcome"):
        if user_input.strip() != "":
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.success(f"Predicted Outcome: {prediction}")
        else:
            st.warning("Enter text first")

# -------------------------------
# Bias Analysis
# -------------------------------
st.subheader("⚖️ Bias Analysis")

# Gender Bias
st.markdown("### Gender vs Outcome")
gender_bias = pd.crosstab(df['gender'], df['outcome'], normalize='index')
st.dataframe(gender_bias)

fig1, ax1 = plt.subplots()
gender_bias.plot(kind='bar', ax=ax1)
ax1.set_title("Gender Bias Distribution")
st.pyplot(fig1)

# Region Bias
st.markdown("### Region vs Outcome")
region_bias = pd.crosstab(df['region'], df['outcome'], normalize='index')
st.dataframe(region_bias)

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
            st.info("No major gender bias observed")
except:
    st.info("Not enough data")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("💡 Built using NLP + Machine Learning | Streamlit App")