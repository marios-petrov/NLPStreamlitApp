import streamlit as st
import nltk
from summarizer import Summarizer,TransformerSummarizer

def BERT_Summarizer(input_text):
    bertModel = Summarizer()
    bertSummary = ''.join(bertModel(input_text,min_length = 60))
    return bertSummary
def GPT_Summarizer(input_text):
    gptModel = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    gptSummary = ''.join(gptModel(input_text,min_length=60))
    return gptSummary
def XLNet_Summarizer(input_text):
    xlnetModel = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    xlnetSummary = ''.join(xlnetModel(input_text, min_length=60))
    return xlnetSummary

def main():
    st.title ("NLP text Processing and Summarization")
    activity1 = ["Summarize","Text Preprocessing"]
    choice = st.sidebar.selectbox("Select Function",activity1)
    if choice == 'summarizer':
        st.subheader("summarizer mode")
        raw_text = st.text_area("Enter Text Here")
        summarizer_choice = st.selectbox("Summary Choice", ["XLNet", "GPT", "BERT"])
        if st.button('Summarizer'):
            if summarizer_choice == "XLNet":
                summarizer_result = XLNet_Summarizer(raw_text)
                st.write(summarizer_result)
            if summarizer_choice == "GPT":
                summarizer_result = GPT_Summarizer(raw_text)
                st.write(summarizer_result)
            if summarizer_choice == "BERT":
                summarizer_result = BERT_Summarizer(raw_text)
                st.write(summarizer_result)





if __name__ == '__main__':
    main()