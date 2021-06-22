import streamlit as st
st.title("DIGIT RECOGNIZER") 

import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2



canvas_result = st_canvas(width=200,height=200,

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3),
stroke_color = st.sidebar.color_picker("Stroke color hex: "),
background_color = st.sidebar.color_picker("Background color hex: ", "#eee"),
#bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")))
realtime_update = st.sidebar.checkbox("Update in realtime", True)



# resize from 250 to 28 x 28
# canvas result.image_data
# convert into usigned int 8
# change to grayscale
# expand dimensions
# model.predict 
# np.argmax
# plot a graph from output


if canvas_result.image_data is not None:
  st.image(canvas_result.image_data)
  img=cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
  rescaled=cv2.resize(img,(150,150),interpolation=cv2.INTER_NEAREST)
  st.write('Rescaled image')
  st.image(rescaled)

from tensorflow import keras
model_new=keras.models.load_model('/content/DIGIT (10).hdf5')
  
if st.button('Predict'):
  img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  op=model_new.predict(img.reshape(1,28,28))
  st.write(f'result:{np.argmax(op[0])}')
  st.bar_chart(op[0])