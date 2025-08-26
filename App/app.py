import streamlit as st
import numpy as np
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3 as inception
from tensorflow.keras.models import Model
import tensorflow.keras.preprocessing.image as tf_image
import pickle
from PIL import Image

encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)


WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
START = "startseq"
STOP = "endseq"
EPOCHS = 10
max_length=25
preprocess_input = inception.preprocess_input

# Load your model
model=load_model('caption_model.h5')


# Load the dictonary
with open('wordtoidx.pkl', 'rb') as f:
    wordtoidx = pickle.load(f)
    
with open('idxtoword.pkl', 'rb') as f:
    idxtoword = pickle.load(f)

# data loding and cleaning
def encodeImage(img):
    img = img.resize((WIDTH, HEIGHT))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x) # Get the encoding vector for the image
    x = np.reshape(x, OUTPUT_DIM )
    return x

def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    print(final)
    st.text_area("",final) 

def response(image_file):
    image=encodeImage(image_file) 
    image = image.reshape((1,OUTPUT_DIM))
    generateCaption(image)


st.markdown(f"""
<style>
    /* Set the background image for the entire app */
    .stApp {{
        background-color:#C0C0C0;
        background-size: 1300px;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    
    </style>
""", unsafe_allow_html=True)

  
# Streamlit interface
st.title("Image Caption")
url='https://tse4.mm.bing.net/th?id=OIP._wKNxe22Elz860LrsTB8SQHaFP&pid=Api&P=0&h=180'
st.image(url,width=700)
uploaded_file = st.file_uploader("Choose a image file",type=["png","jpg","jpeg"])
if uploaded_file is not None:
        image = tf_image.load_img(uploaded_file, target_size=(HEIGHT, WIDTH))
        st.write(image)
        
# Button to classify
if st.button("Show The Image Caption "):
    try:
        response(image)
    except Exception as e:
        predicted_word=("error occured : ",e)
        st.text_area("",)
