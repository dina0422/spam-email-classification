import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Classifier")
st.write("Enter the content of an email below to check if it's spam or not.")

# User input
email_input = st.text_area("Email Content", height=200)

# Predict button
if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        email_features = vectorizer.transform([email_input])
        prediction = model.predict(email_features)[0]

        if prediction == 1:
            st.error("🚨 This email is likely **SPAM**!")
        else:
            st.success("✅ This email is **Not Spam**.")

