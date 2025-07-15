# importing packages
import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 
import nltk
import re
nltk.download('wordnet') 
nltk.download("stopwords")
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# creating object
le=WordNetLemmatizer()

# loading saved models
with open("bag_of_word.pkl","rb")as file: 
    bag_of_word=pickle.load(file)

with open("final_model.pkl","rb")as file: 
    model=pickle.load(file)

# title
st.title("Spam message detection")
# prompt for enter message
input_message=st.text_input(label="please enter your message")

# data preprocessing function
def preprocessing(data): 
    clean_data=re.sub("[^a-zA-Z]"," ",data) 
    lower_data=clean_data.lower()
    tokenize=lower_data.split()
    stemming=[le.lemmatize(word) for word in tokenize if not word in stopwords.words("english")]
    processed_data=" ".join(stemming)
    return processed_data 

# output button
if st.button("spam detection"): 
    processed_message=preprocessing(input_message)

    bow_data=bag_of_word.transform([processed_message]).toarray()
    prediction=model.predict(bow_data) 
    if prediction == 1: 
        st.warning("warning spam message")
    else: 
        st.success("message is not spam")
