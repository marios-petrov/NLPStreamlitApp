import streamlit as st
from io import StringIO
from bs4 import BeautifulSoup
import requests
import re
from summarizer import Summarizer,TransformerSummarizer
import nltk
import string
from collections import Counter
from nltk.util import ngrams

#These three following functions are my NLP summarizers implemented using the Summarizer library (XLNet performs the best)
def BERT_Summarizer(input_text):
    bertModel = Summarizer()
    bertSummary = ''.join(bertModel(input_text,min_length=50))
    return bertSummary

def GPT_Summarizer(input_text):
    gptModel = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    gptSummary = ''.join(gptModel(input_text,min_length=50))
    return gptSummary

def XLNet_Summarizer(input_text):
    xlnetModel = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    xlnetSummary = ''.join(xlnetModel(input_text,min_length=50))
    return xlnetSummary

#This is a helper function that generates the summary from the input and summary of choice
def summarizer_result (raw_text, summarizer_choice):
    if summarizer_choice == "XLNet":
        summarizer_result = XLNet_Summarizer(raw_text)
        return summarizer_result
    if summarizer_choice == "GPT":
        summarizer_result = GPT_Summarizer(raw_text)
        return summarizer_result
    if summarizer_choice == "BERT":
        summarizer_result = BERT_Summarizer(raw_text)
        return summarizer_result

#This iis my wikipedia scraper that I used BeautifulSoup to implement
def wikiScrapper(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text,"xml")
    paragraphs = soup.findAll("p")
    for i in range(len(paragraphs)):
        paragraphs[i] = paragraphs[i].get_text()
        pattern = r"\[[^\]]*\]"
    for i in range(len(paragraphs)):
        paragraphs[i] = re.sub(pattern, "", paragraphs[i])
        text = "".join(paragraphs)
        return text

#Main function where the magic happens
def main():
    st.title ("NLP text Processing and Summarization")
    activityOptions = ["Summarization","Text Processing"]
    selectActivity = st.sidebar.selectbox("Select Function",activityOptions)

    if selectActivity == 'Summarization':
        st.subheader("Summarizer mode")
        inputFormat = st.radio('Choose Input Format', ('TextFile', 'RawText', 'WikiURL'))
        summarizer_choice = st.selectbox("Summary Choice", ["XLNet", "GPT", "BERT"])

        if inputFormat == 'TextFile':
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file is not None:
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    string_data = stringio.read()
                    if st.button('Summarize'):
                       st.write(summarizer_result(string_data, summarizer_choice))

        if inputFormat == 'RawText':
                raw_text = st.text_area("Enter Text Here")
                if st.button('Summarize'):
                   st.write(summarizer_result(raw_text, summarizer_choice))

        if inputFormat == 'WikiURL':
               urlInput=st.text_input("Enter URL link")
               if st.button('Summarize'):
                  wikiTextContent = wikiScrapper(urlInput)
                  st.write(summarizer_result(wikiTextContent, summarizer_choice))

    if selectActivity == 'Text Processing':
        st.subheader("Text Processor mode")
        inputFormat = st.radio('Choose Input Format', ('TextFile', 'RawText'))
        processingFunction= st.radio('Choose Text Processing Function', ('Remove ALL Punctuation', 'Remove Stop Words', 'Tag all Parts-of-Speech(POS)', 'Count n-grams', 'Print Most Common Words'))

        if inputFormat == 'TextFile':
            st.markdown('*Only .txt files accepted')
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                string_data = uploaded_file.read().decode('utf-8')
                uploaded_file.close()

                if processingFunction == 'Remove ALL Punctuation':
                    noPunctuation = string_data.translate(str.maketrans('','',string.punctuation))
                    tokenizedText = nltk.word_tokenize(noPunctuation)
                    stringContent = str(tokenizedText)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

                if processingFunction == 'Remove Stop Words':
                    tokenizedText = nltk.word_tokenize(string_data)
                    filteredList = []
                    stopWordsList = nltk.corpus.stopwords.words('english')
                    for w in tokenizedText:
                        if w.lower() not in stopWordsList:
                            filteredList.append(w)
                    stringContent = str(filteredList)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

                if processingFunction == 'Tag all Parts-of-Speech(POS)':
                    tokenizedText = nltk.word_tokenize(string_data)
                    posTagger = nltk.pos_tag(tokenizedText)
                    stringContent = str(posTagger)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

                if processingFunction == 'Count n-grams':
                    numberofNgrams = st.radio('n-gram size',('unigram','bigram'))
                    tokenizedText = nltk.word_tokenize(string_data)
                    
                    if numberofNgrams == 'unigram':
                        unigrams = ngrams(tokenizedText,1)
                        unigramCount = Counter(unigrams)
                        stringContent = str(unigramCount)
                        st.download_button(
                            label='Download Result',
                            data=stringContent,
                            file_name='Processed File', mime="text")
                    if numberofNgrams == 'bigram':
                        bigrams = ngrams(tokenizedText,2)
                        bigramsCount = Counter(bigrams)
                        stringContent = str(bigramsCount)
                        st.download_button(
                            label='Download Result',
                            data=stringContent,
                            file_name='Processed File', mime="text")

                if processingFunction == 'Print Most Common Words':
                    tokenizedText = nltk.word_tokenize(string_data)
                    mostCommonWords = Counter(tokenizedText).most_common()
                    stringContent = str(mostCommonWords)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")



        if inputFormat == 'RawText':
            raw_text = st.text_area("Enter Text Here")
            st.markdown('*All text processing automatically tokenizes the data as well')

            if processingFunction == 'Remove ALL Punctuation':
                noPunctuation = raw_text.translate(str.maketrans('','',string.punctuation))
                tokenizedText = nltk.word_tokenize(noPunctuation)
                if st.button('Process'):
                    stringContent = str(tokenizedText)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

            if processingFunction == 'Remove Stop Words':
                tokenizedText = nltk.word_tokenize(raw_text)
                filteredList = []
                stopWordsList = nltk.corpus.stopwords.words('english')
                for w in tokenizedText:
                    if w.lower() not in stopWordsList:
                        filteredList.append(w)
                if st.button('Process'):
                    stringContent = str(filteredList)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

            if processingFunction == 'Tag all Parts-of-Speech(POS)':
                tokenizedText = nltk.word_tokenize(raw_text)
                posTagger = nltk.pos_tag(tokenizedText)
                if st.button('Process'):
                    stringContent = str(posTagger)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")

            if processingFunction == 'Count n-grams':
                numberofNgrams = st.radio('n-gram size',('unigram','bigram'))
                tokenizedText = nltk.word_tokenize(raw_text)
                
                if numberofNgrams == 'unigram':
                    unigrams = ngrams(tokenizedText,1)
                    unigramCount = Counter(unigrams)
                    if st.button('Process'):
                        stringContent = str(unigramCount)
                        st.download_button(
                            label='Download Result',
                            data=stringContent,
                            file_name='Processed File', mime="text")
                if numberofNgrams == 'bigram':
                    bigrams = ngrams(tokenizedText,2)
                    bigramsCount = Counter(bigrams)
                    if st.button ('Process'):
                        stringContent = str(bigramsCount)
                        st.download_button(
                            label='Download Result',
                            data=stringContent,
                            file_name='Processed File', mime="text")

            if processingFunction == 'Print Most Common Words':
                tokenizedText = nltk.word_tokenize(raw_text)
                mostCommonWords = Counter(tokenizedText).most_common()
                if st.button('Process'):
                    stringContent = str(mostCommonWords)
                    st.download_button(
                        label='Download Result',
                        data=stringContent,
                        file_name='Processed File', mime="text")




if __name__ == '__main__':
    main()
