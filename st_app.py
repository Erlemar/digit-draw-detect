import logging

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.ml_utils import predict, get_model, transforms
from src.utils import plot_img_with_rects, save_image

st.title('Handwritten digit detector')
logging.info('Starting')

col1, col2 = st.columns(2)

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color='#fff',
        stroke_width=5,
        stroke_color='#000',
        background_color='#fff',
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode='freedraw',
        key='canvas',
    )
with col2:
    logging.info('canvas ready')
    if canvas_result.image_data is not None:
        # convert a drawn image into numpy array with RGB from a canvas image with RGBA
        img = np.array(Image.fromarray(np.uint8(canvas_result.image_data)).convert('RGB'))
        image = transforms(image=img)['image']
        logging.info('image augmented')
        model = get_model()
        logging.info('model ready')
        pred = predict(model, image)
        logging.info('prediction done')

        file_name = save_image(image.permute(1, 2, 0).numpy(), pred)
        threshold = st.slider('Bbox probability slider', min_value=0.0, max_value=1.0, value=0.8)

        fig = plot_img_with_rects(image.permute(1, 2, 0).numpy(), pred, threshold, coef=192)
        fig.savefig(f'{file_name}_temp.png')
        image = Image.open(f'{file_name}_temp.png')
        st.image(image)

text = """
This is a small app for handwritten digit recognition and recognition developed for fun. It uses a handwritten YOLOv3 model trained from scratch.
You can draw a digit (or whatever you want) and the model will try to understand what is it.
You can use the slider above to show bounding boxes with a probability higher than the threshold.
If you want to know how the app works in more detail, you are welcome to read "About" page.
Enjoy! :)
"""

st.markdown(text, unsafe_allow_html=True)
