from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = '6110e954da468910321cb1d044259f31'

# Load the trained Random Forest model and TfidfVectorizer
model_path = 'best_rf_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
rf_model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Preprocess the text data
def preprocess_input(text):
    # Lowercase the text
    text = text.lower()
    return text

# Function to predict sentiment
def predict_sentiment(review):
    preprocessed_review = preprocess_input(review)
    tfidf_review = vectorizer.transform([preprocessed_review])
    decision_scores = rf_model.predict_proba(tfidf_review)[0]

    # Find the index of the class with the highest probability
    predicted_class = decision_scores.argmax()

    # Map the index to the corresponding sentiment class
    sentiment_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}
    sentiment = sentiment_mapping[predicted_class]

    return sentiment


class ReviewForm(FlaskForm):
    review = TextAreaField('Enter your product review:', validators=[DataRequired()])
    submit = SubmitField('Predict Sentiment')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = ReviewForm()
    result = None
    if form.validate_on_submit():
        user_input = form.review.data
        sentiment = predict_sentiment(user_input)
        result = f"Predicted sentiment: {sentiment}"
    return render_template('index.html', form=form, result=result)

if __name__ == "__main__":
    app.run(debug=True)
