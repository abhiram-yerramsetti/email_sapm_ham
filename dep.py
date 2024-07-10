import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the pre-trained model
model = pickle.load(open(model.pkl", 'rb'))

# Load the CountVectorizer used for training
with open("bow.pkl", 'rb') as f:
    bow = pickle.load(f)

st.image("logo.png",width=200)
st.title("Email Spam/Ham Classifier")

# Input email text
Email = st.text_input("Paste the email here:")

# Check if the email input is not empty
if Email:
    # Transform the input email text to feature array
    data = bow.transform([Email]).toarray()

    # Predict if the email is spam or ham
    spam_ham = model.predict(data)[0]

    # Display the prediction when the button is pressed
    if st.button('Submit'):
        st.write("The email is:", "Spam" if spam_ham == 'spam' else "Ham")
        if spam_ham == 'spam':
            st.image("OIP.jpeg",width=200)
        else :
            st.image("ham.jpeg",width=200)
