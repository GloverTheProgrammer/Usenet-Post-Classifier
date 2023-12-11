import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import nltk
from nltk.corpus import stopwords
import string

#Load Model
def load_model():
    model_path = 'multinomial_nb_model.joblib'
    return load(model_path)

# Load Feature Names
def load_feature_names():
    feature_names_df = pd.read_csv('feature_names.csv')
    return feature_names_df['title'].tolist()

# Preprocess Text Input by User exactly like the training data (Bag of Words)
def preprocess_text(text, feature_names):
    words = text.split()
    words = [word.lower() for word in words if len(word) > 2 and word.lower() not in stop_words]
    features = np.zeros(len(feature_names))
    word_counts = {word: words.count(word) for word in words}
    for i, word in enumerate(feature_names):
        features[i] = word_counts.get(word, 0)
    return features.reshape(1, -1)

# Load Target Names
def load_target_names():
    return np.load('target_names.npy', allow_pickle=True)

# Load Stop Words
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(string.punctuation)


# Load Model, Feature Names, and Target Names
model = load_model()
feature_names = load_feature_names()
target_names = load_target_names()

# Streamlit Layout
st.title('Usenet Post Classifier')

with st.sidebar:
    st.header("Settings")
    prediction_option = st.selectbox("Number of Predictions", ['Top Prediction', 'Top 3'])

st.markdown("""
This tool classifies Usenet posts into categories. Please follow the format below for the input:
""")

st.markdown("""
**Input Format:**
- Start with the `Path:` followed by the network path.
- Include `From:`, `Newsgroups:`, `Subject:`, `Date:`, `Organization:`, `Lines:`, `Message-ID:`, `NNTP-Posting-Host:`.
- Write the content of the post after these headers.

**Example:**
```
Path: cantaloupe.srv.cs.cmu.edu!crabapple.srv.cs.cmu.edu!fs7.ece.cmu.edu...
From: user@example.com (User Name)
Newsgroups: comp.windows.x
Subject: Query about X Window System
Date: 1 Jan 1993 12:00:00 GMT
Organization: Example University
Lines: 15
Message-ID: <1234abc@news.example.edu>
NNTP-Posting-Host: example.edu

Content of the post goes here...
```
""")


# Take in User Input and Classify
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Enter your text below:", height=350)
with col2:
    st.write("") 
    if st.button("Classify"):
         if user_input:
            with st.spinner("Classifying..."):
                processed_input = preprocess_text(user_input, feature_names) 
                probabilities = model.predict_proba(processed_input)[0]
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_classes = [target_names[i] for i in top_indices]
                top_probabilities = probabilities[top_indices]

                if prediction_option == 'Top Prediction':
                    st.write(f"Top Prediction: {top_classes[0]} with a probability of {top_probabilities[0]:.2f}")
                else:
                    st.write("Top 3 Predictions:")
                    for i in range(3):
                        st.write(f"{i+1}: {top_classes[i]} with a probability of {top_probabilities[i]:.2f}")
            st.success("Classification Done!")
    else:
        st.error("Please enter text to classify.")





   