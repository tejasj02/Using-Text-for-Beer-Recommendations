import streamlit as st
from scripts.inference import predict_beer

st.set_page_config(page_title="Beer Recommender", page_icon="🍺")
st.title("🍻 What Beer Should I Have?")
st.write("Enter a review-style description and I'll give you a beer!")

# Text input
custom_review = st.text_area("Describe the beer:", height=150)

if st.button("Predict Beer"):
    if custom_review.strip():
        with st.spinner("Analyzing..."):
            try:
                result = predict_beer(custom_review)
                st.success(f"🍺 Predicted Beer: **{result}**")
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    else:
        st.warning("Please enter a description first.")
