from flask import Flask, request, render_template,json,jsonify
import pickle
import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import nltk
import re

nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the model and vectorizer from pickle files
model = joblib.load("sentmodel.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess the text
        X = vectorizer.transform([text])
        # Make a predictione
        prediction = model.predict(X)
        # Return the prediction
        return render_template('result.html', prediction=prediction)
    return render_template('home.html')


def predict_sentiment_text(text):
    # Preprocess the text
    X = vectorizer.transform([text])
    # Make a prediction
    prediction = model.predict(X)[0]
    return prediction




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True)