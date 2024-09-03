import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import numpy as np

# Load pre-trained T5 model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

# Move model to device (GPU or CPU)
model.to(device)

# Function to generate product description
def generate_description(product_name, product_category, product_keywords):
    input_text = f"Generate a product description for a {product_category} product called {product_name} with keywords {product_keywords}."
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output = model.generate(input_ids, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app
st.title("Product Description Generator")

# Form to input product information
with st.form("product_info"):
    product_name = st.text_input("Product Name")
    product_category = st.selectbox("Product Category", ["Electronics", "Fashion", "Home Goods", "Other"])
    product_keywords = st.text_input("Product Keywords (comma separated)")
    submit_button = st.form_submit_button("Generate Description")

# Generate and display product description
if submit_button:
    description = generate_description(product_name, product_category, product_keywords)
    st.write(description)
