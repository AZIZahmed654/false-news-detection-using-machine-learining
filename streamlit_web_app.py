import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

# Load the model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model2.pkl', 'rb') as model_lr_file:
    model_lr = pickle.load(model_lr_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

class Preprocessing:
    def __init__(self, data):
        self.data = data
        
    def text_preprocessing_user(self):
        lm = WordNetLemmatizer()
        pred_data = [self.data]    
        preprocess_data = []
        for data in pred_data:
            review = re.sub('[^a-zA-Z0-9]', ' ', data)
            review = review.lower()
            review = review.split()
            review = [lm.lemmatize(x) for x in review if x not in stopwords_set]
            review = " ".join(review)
            preprocess_data.append(review)
        return preprocess_data

def predict_news(news_text):
    preprocess_data = Preprocessing(news_text).text_preprocessing_user()
    data = tfidf_vectorizer.transform(preprocess_data)
    prediction = model.predict(data)
    return "The News Is Fake" if prediction[0] == 0 else "The News Is Real"

def main():
    st.title('False News Detection App')


      # Model selection
    model_choice = st.radio("Select Model:", ('Random Forest', 'Logistic Regression'))

    # Choose the model based on user selection
    if model_choice == 'Random Forest':
        selected_model = model
    else:
        selected_model = model_lr

    # Get user input
    news_text = st.text_area('Enter the news text:', '')

    # Make prediction when the user clicks the button
    if st.button('Predict'):
        result = predict_news(news_text)
        st.success(result)

if __name__ == '__main__':
    main()
