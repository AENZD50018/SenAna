import joblib

# Load the trained SVM model and vectorizer
model_path = 'svm_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
svm_model = joblib.load(model_path)
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
    prediction = svm_model.predict(tfidf_review)[0]
    return "Positive" if prediction == 1 else "Negative"

def main():
    print("Welcome to Product Review Sentiment Analysis!")
    print("Enter 'exit' to quit the program.")
    while True:
        user_input = input("Enter your product review: ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        sentiment = predict_sentiment(user_input)
        print(f"Predicted sentiment: {sentiment}\n")

if __name__ == "__main__":
    main()
