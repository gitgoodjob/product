import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from PIL import Image
import numpy as np

# Load pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to generate product description
def generate_description(product_name, product_category, product_keywords):
    input_text = f"Generate a product description for a {product_category} product called {product_name} with keywords {product_keywords}."
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app
st.title("Product Description Generator")

# Create a form to input product information
with st.form("product_info"):
    product_name = st.text_input("Product Name")
    product_category = st.selectbox("Product Category", ["Electronics", "Fashion", "Home Goods", "Other"])
    product_keywords = st.text_input("Product Keywords (comma separated)")
    submit_button = st.form_submit_button("Generate Description")

# Generate and display product description
if submit_button:
    description = generate_description(product_name, product_category, product_keywords)
    st.write(description)

# Add some aesthetic elements to the app
st.image("https://picsum.photos/200/300", width=200)
st.markdown("### Product Description Generator")
st.write("This app uses a pre-trained T5 model to generate product descriptions based on the product name, category, and keywords.")
