# sms-spam-classifier
SMS Spam Classifier using NLP, TF-IDF, and Multinomial Naive Bayes (98% accuracy)
A Machine Learning model that classifies SMS messages as Spam or Ham (Not Spam) using NLP preprocessing, TF-IDF vectorization, and Multinomial Naive Bayes.
Achieves ~98% accuracy on the UCI SMS Spam Collection dataset.
Project Overview

This project builds an end-to-end SMS spam detection system using:

Text preprocessing (NLP)

TF-IDF feature extraction

Supervised machine learning

Model evaluation & metrics

Saving model for deployment
The goal is to classify messages such as:
"Congratulations! You won a lottery!" → SPAM  
"Hey, are we meeting tomorrow?" → HAM
UCI Machine Learning Repository – SMS Spam Collection

Total messages: 5,574

Labels: spam or ham

Link: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
Rename the dataset file to: spam.csv
df = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])
Technologies Used

Python

NLTK

scikit-learn

Pandas / NumPy

Matplotlib / Seaborn
NLP Preprocessing Pipeline

The text messages were cleaned using:

Lowercasing

Removing URLs

Removing HTML tags

Removing punctuation

Removing numbers

Tokenization

Removing stopwords

Lemmatization

Joining back cleaned text
Model Building
TF-IDF Vectorization
Converted text into numerical features using:TfidfVectorizer(max_features=3000)
TRAIN/Test split train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
Model Used: Multinomial Naive Bayes

Chosen because it performs extremely well on text data.
Model Performance

Achieved:

Accuracy: ~98%

High spam precision & recall

Clean confusion matrix

Strong generalization on test data

Example:

Accuracy: 0.9793 (97.93%)

 Saving the Model

The trained model and TF-IDF vectorizer were saved:

pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))Prediction Function

Example of predicting new SMS messages:

def predict_sms(text):
    cleaned = preprocess(text)
    vector = tfidf.transform([cleaned]).toarray()
    prediction = model.predict(vector)[0]
    return "SPAM" if prediction == 1 else "HAM"
    PROJECT STRUCTURE
    sms-spam-classifier/
│── sms_spam_classifier.ipynb
│── spam.csv
│── spam_model.pkl
│── vectorizer.pkl
│── requirements.txt
│── README.md
Author

Nandini
Machine Learning & Data Science Enthusiast
