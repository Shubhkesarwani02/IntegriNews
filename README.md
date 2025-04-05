# 📰 Fake News Detection App

A simple machine learning-based web app built using **Streamlit** that detects whether a news article is **Fake** or **Real**. The model is trained using natural language processing (NLP) techniques and classical machine learning algorithms.

---

## 🚀 Features

- Input any news headline or content
- Detects whether the input is fake or real
- Trained using **TF-IDF vectorization** and **Logistic Regression**
- Deployable via **Streamlit Cloud**

---

## 📁 Dataset

Used a cleaned and balanced dataset of 40,000+ news articles with binary labels:  
`Fake` (0) or `Real` (1)

---

## 🧠 Model

**Preprocessing Steps**:
- Lowercasing text
- Removing stopwords, punctuation, and special characters
- Tokenization and Lemmatization (using `nltk`)

**Vectorization**:
- `TF-IDF` with top 5000 features

**Model**:
- Logistic Regression (`class_weight='balanced'`)

---

## 📦 Libraries Used

- numpy  
- pandas  
- nltk  
- scikit-learn  
- joblib  
- matplotlib  
- re  
- streamlit  

---

## 🧪 Evaluation

- Accuracy: ~90% on the validation set
- Other models tested: `SVM`, `Naive Bayes`
- Final model chosen: `Logistic Regression` (best trade-off between performance and speed)

---

## 💾 Saved Files

- `Fake_News_Model.pkl` – Trained Logistic Regression model  
- `tfidf_vectorizer.pkl` – TF-IDF vectorizer used in preprocessing

---

## 🖥️ Deployment

App is deployed using **Streamlit Cloud**.

### To run locally:

```bash
git clone https://github.com/your-username/FakeNewsDetector.git
cd FakeNewsDetector
pip install -r requirements.txt
streamlit run app.py
