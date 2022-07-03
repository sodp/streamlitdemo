#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pandas as pd
import cv2
from PIL import Image , ImageOps
import numpy as np


# In[2]:


#Allow cache in the application for prediction
#@st.cache(allow_output_mutation = True, hash_funcs={builtins.function: my_hash_func})

#Load deep learning model
def load_model():
    model = tf.keras.models.load_model('C:/Users/Siddhant.Panda/Desktop/app/new_trained.h5')
    return model
with st.spinner('Model is being Loaded...'):
    model = load_model()

    #sample comment to stash
# In[3]:


st.markdown("<h1 style = 'text-align: center;'>Covid/Pneumonia Detection Using Chest X-Ray Images</h1>", unsafe_allow_html=True)
instructions = """
                Please Upload Your Chest X-Ray Images Here
                The image you select or upload will be fed through the
                Deep Neural Network in real-time and the output will be displayed to the screen.
                """
st.write(instructions)
#file uploader with multiple upload option
file = st.file_uploader("Upload the image to be classified \U0001F447" , type=["jpg","png","jpeg"],accept_multiple_files=True)
st.set_option('deprecation.showfileUploaderEncoding',False)


# In[4]:


#Function to pre-process the uploaded image and return prediction
def upload_predict(upload_image,model):
    size =(128,128)
    image = ImageOps.fit(upload_image,size,Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis ,...]
    prediction  = model.predict(img_reshape)
    return prediction


# In[5]:


if file is None:
    st.text("Please upload an image file")
else:
    for img_upload in (file):
        st.markdown("<h1 style ='text-align: center;'>Here is the image you selected</h1>" ,unsafe_allow_html = True)
        image =Image.open(img_upload)
        st.image(image , use_column_width = True , caption = img_upload.name)
        predictions = upload_predict(image , model)
        score = tf.nn.softmax(predictions[0])
        
        st.title("Here are the results")
        st.write("Predictions:")
        predictions_op = pd.DataFrame(predictions , columns =['Covid','Normal','Pneumonia'])
        st.wrtie(predictions_op)
        st.write('Percentage Confidence:')
        st.write(score)
        class_names = ['Covid','Normal' ,'Pneumonia']
        st.write("This image most likely belongs to {} category with a {:.2f} percent confidence".format(class_names[np.argmax(score)],100*np.max(score)))
        print("This image most likely belongs to {} category with a {:.2f} percent confidence.".format(class_names[np.argmax(score)],100*np.max(score)))
        


# In[ ]:




