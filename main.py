import streamlit as st
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
import keras.preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Apollo")

uploaded_file = st.file_uploader("Pick a file")
model = load_model("model.h5")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = image.resize((150, 150))
    #img = keras.preprocessing.image.load_img(image, target_size=(150, 150))
    img_tensor = keras.preprocessing.image.img_to_array(image)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.
    # x = np.true_divide(x, 255)

    pred = model.predict(img_tensor)

    if pred>0.9:
        st.write('Your image has been classified. There is no visible tumor in your MRI scan.')
        st.write('Accuracy: ')
        st.write(pred[0][0]*100)
    if pred<0.05:
        st.write('Your image has been classified. Your MRI scan contains a tumor.')
