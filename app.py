import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('C:/Users/Admin/Desktop/email spam classification/model.pkl', 'rb'))

# Define the function for preprocessing text
def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters and numbers
    words = text.split()  # Tokenize text
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    words = [ps.stem(word) for word in words]  # Stem words
    return ' '.join(words)  # Join words back into a single string

# Streamlit app
st.title("Email Spam Classifier")
input_sms = st.text_area("Enter message")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

