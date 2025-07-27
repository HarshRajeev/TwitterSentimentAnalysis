# Twitter Sentiment Analysis using Machine Learning

This project analyzes sentiments of real-time tweets using a trained machine learning model. It classifies tweets into **positive** or **negative** categories based on text content.

## 🔍 Features

- Preprocessing using:
  - Stemming (`PorterStemmer`)
  - Stopword removal (`nltk`)
- Feature extraction using `TfidfVectorizer`
- Sentiment prediction using:
  - Logistic Regression / XGBoost / SVM / Naive Bayes 

---

## 📁 Project Structure

```

├── sentiment\_model.pkl         # Trained ML model (Logistic Regression/XGBoost)
├── tfidf\_vectorizer.pkl        # Trained TF-IDF vectorizer
├── twitter\_sentiment.py       # Main script for fetching & analyzing tweets
├── train\_model.py             # Script to train & save model and vectorizer
├── requirements.txt
└── README.md

````

---

## 📦 Installation

1. **Clone this repository**:
```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
````

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **NLTK setup** (only once):

```python
import nltk
nltk.download('stopwords')
```

---

## 🧠 Model Training

If you haven’t created the model and vectorizer yet, run:

```bash
python train_model.py
```

This will generate:

* `sentiment_model.pkl`
* `tfidf_vectorizer.pkl`

Make sure your dataset is loaded and preprocessed similarly to real tweet inputs (e.g., stemmed, cleaned).

---


#Comparison between diffrent models
| Model                 | Accuracy | Speed  | 
|-----------------------|----------|--------|
| Logistic Regression   | 80.66%   | Fast   | 
| XG Boost              | 74.50%   | Medium | 
| Linear SVM            | 76.96%   | Fast   | 
| Naive Bayes           | 75.58%   | Fast   | 
---


## 🛠 Built With

* Python
* Scikit-learn
* Logistic Regression and XGBoost (optional)
* NLTK (for preprocessing)

---
