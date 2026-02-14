import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

###load the imdb data set word index
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

### load the model file


model=load_model("simple_rnn_imdb.h5")


### helper functions
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i -3 ,'?') for i in encoded_review])


def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


### prediction function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment= "Positive" if prediction[0][0]>0.5 else "Negative"

    return sentiment,prediction[0][0]



### title 

st.title("Imdb movie review sentiment analysis")
st.write("enter a movie review to classify whether the review is positive or negative")


### user input

user_input=st.text_area("movie review")

if st.button("classify"):
    preprocessed_input=preprocess_text(user_input)
    
    ###make prediction
    prediction=model.predict(preprocessed_input)
    sentiment="Positive" if prediction[0][0]>0.5 else "Negative"


    ##display the result
    st.write(f'sentiment:{sentiment}')
    st.write(f'prediction score: {prediction[0][0]}')

else:
    st.write("please enter a movie review")