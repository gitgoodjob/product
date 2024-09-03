import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load pre-trained GPT-J model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForSeq2SeqLM.from_pretrained("EleutherAI/gpt-j-6B")

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
