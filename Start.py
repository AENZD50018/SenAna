import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset_path = 'SA.csv'
df = pd.read_csv(dataset_path)

# Preprocess the text data
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    return text

df['Review'] = df['Review'].apply(preprocess_text)

# Split the data into features (X) and labels (y)
X = df['Review']
y = df['Sentiment']

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=2)
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Model
rf_model = RandomForestClassifier(random_state=42)

# Hyperparameter Tuning using Grid Search
param_grid = {
    'n_estimators': [50, 100, 200],   # Number of trees in the forest
    'max_depth': [None, 10, 20],      # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
}

grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize the Random Forest model with the best hyperparameters
best_rf_model = RandomForestClassifier(random_state=42, **best_params)

# Train the model on the training set with the best hyperparameters
best_rf_model.fit(X_train, y_train)

# Evaluate the best model using cross-validation
cv_accuracy = cross_val_score(best_rf_model, X_tfidf, y, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy:", cv_accuracy.mean())

# Evaluate the model on the test set
y_pred = best_rf_model.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))



