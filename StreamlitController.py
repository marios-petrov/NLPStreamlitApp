import streamlit as st
import pandas as pd
#import WikiBERT as bert
#import WikiScrapper as ws

input_type = st.radio("Input Type: ", ["URL", "Raw Text"])
st.markdown("<h3 style='text-align: center;'>Input</h3>", unsafe_allow_html=True)

if input_type == "Raw Text":
    with open("raw_data/input.txt") as f:
        sample_text = f.read()
    text = st.text_area("", sample_text, 200)
else:
    url = st.text_input("")


