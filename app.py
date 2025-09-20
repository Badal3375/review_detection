import streamlit as st
import pickle

# --- Load Model & Vectorizer ---
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# --- Streamlit UI ---
st.title(" Reviews / Sentiment Detection")
st.write("Enter a review/news text and find out if it is Positive or Negative .")

# Input text
user_input = st.text_area("Enter your review/news here:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform input
        transformed_input = vectorizer.transform([user_input])

        # Prediction
        prediction = model.predict(transformed_input)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This review is **Positive / Real**")
        else:
            st.error("❌ This review is **Negative / Fake**")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
