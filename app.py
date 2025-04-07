import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("Fake_News_Model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_news(text):
    """Predict whether the news is Fake or Real and display the confidence score."""
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    confidence_score = model.predict_proba(vector)[0][prediction]
    result = "Fake" if prediction == 0 else "Real"
    return result, confidence_score

# Add Background using CSS
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://source.unsplash.com/1600x900/?news");
            background-size: cover;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_url()

# Streamlit App
st.title("📰 IntegriNews - Fake News Detector")
st.write("Enter a news headline or article to check if it's Fake or Real.")

# Sidebar for Additional Information
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This Fake News Detector uses a Logistic Regression model trained with TF-IDF vectorization."
    " It analyzes the text and predicts whether it's Fake or Real with a confidence score."
)
st.sidebar.write("🔎 Enter a news article in the main panel to check for accuracy.")

# Text Input
user_input = st.text_area("Enter News Text:", "")

# Prediction Button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        result, confidence_score = predict_news(user_input)
        if result == "Fake":
            st.error(f"The news is **{result}** 🛑")
        else:
            st.success(f"The news is **{result}** ✅")
        st.write(f"📊 Confidence Score: **{confidence_score:.2%}**")

# Clear Button
if st.button("Clear"):
    st.experimental_rerun()

# Footer
st.caption("Built with Streamlit")
