import joblib

def load_model_and_vectorizer():

    model = joblib.load("spam_classifier_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer