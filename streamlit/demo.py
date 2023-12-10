import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import nltk
from nltk.corpus import stopwords
import string

@st.cache_resource
def load_model():
    model_path = 'multinomial_nb_model.joblib'
    return load(model_path)


@st.cache_data
def load_feature_names():
    feature_names_df = pd.read_csv('feature_names.csv')
    return feature_names_df['word'].tolist()


nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(string.punctuation)


def preprocess_text(text, feature_names):
    words = text.split()
    # Remove stopwords and non-feature words, count word occurrences
    word_counts = {word: words.count(word) for word in set(words) if word in feature_names and word not in stop_words}
    # Transform to feature vector
    features = np.zeros(len(feature_names))
    for i, word in enumerate(feature_names):
        features[i] = word_counts.get(word, 0)
    return features


model = load_model()
feature_names = load_feature_names()


st.title("Text Classification Demo")

user_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if user_input:
        processed_input = preprocess_text(user_input, feature_names)
        prediction = model.predict([processed_input])
        st.write(f"Predicted Class: {prediction[0]}")
    else:
        st.error("Please enter some text to classify.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as string:
    string_data = uploaded_file.getvalue()

    # or to read file into a dataframe (e.g., if it's a CSV):
    import pandas as pd
    df = pd.read_csv(uploaded_file)