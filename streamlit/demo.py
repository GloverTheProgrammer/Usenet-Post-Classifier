import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import nltk
from nltk.corpus import stopwords
import string


def load_model():
    model_path = 'multinomial_nb_model.joblib'
    return load(model_path)



def load_feature_names():
    feature_names_df = pd.read_csv('feature_names.csv')
    return feature_names_df['title'].tolist()


def preprocess_text(text, feature_names):
    # Tokenize and remove stopwords and punctuation
    words = text.split()
    # Filter out short words and stop words, then lowercase
    words = [word.lower() for word in words if len(word) > 2 and word.lower() not in stop_words]
    # Initialize feature vector based on top 2000 words
    features = np.zeros(len(feature_names))
    # Count occurrences of each word in the top 2000 words
    word_counts = {word: words.count(word) for word in words}
    for i, word in enumerate(feature_names):
        features[i] = word_counts.get(word, 0)
    return features.reshape(1, -1)



def load_target_names():
    return np.load('target_names.npy', allow_pickle=True)


nltk.download('stopwords')
stop_words = set(stopwords.words('english')) | set(string.punctuation)



model = load_model()
feature_names = load_feature_names()
target_names = load_target_names()


st.title("Newsgroup Classification App")
user_input = st.text_area("This app uses a machine learning model to classify text. Enter your text below and press 'Classify")
prediction_option = st.selectbox("How many predictions would you like to see?", ('Top Prediction', 'Top 3'))

if st.button("Classify"):
    if user_input:
        with st.spinner("Classifying..."):
            processed_input = preprocess_text(user_input, feature_names) # Ensure this function is defined
            probabilities = model.predict_proba(processed_input)[0]
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_classes = [target_names[i] for i in top_indices]
            top_probabilities = probabilities[top_indices]

            # Display predictions based on user choice
            if prediction_option == 'Top Prediction':
                st.write(f"Top Prediction: {top_classes[0]} with a probability of {top_probabilities[0]:.2f}")
            else:
                st.write("Top 3 Predictions:")
                for i in range(3):
                    st.write(f"{i+1}: {top_classes[i]} with a probability of {top_probabilities[i]:.2f}")
        st.success("Classification Done!")
    else:
        st.error("Please enter some text to classify.")