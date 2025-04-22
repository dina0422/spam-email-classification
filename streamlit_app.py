import streamlit as st
import joblib

model = joblib.load("spam_classifier_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

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

