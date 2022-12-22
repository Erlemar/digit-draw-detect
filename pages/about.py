import streamlit as st

text = """
## A brief history of the project.
The first idea of the project was conceived in the summer of 2017. I completed cs231n course and wanted to put my skills to the test. So I studied Flask and many other tools from scratch and made an [app](https://github.com/Erlemar/digit-draw-recognize) for recognizing handwritten digits. It had two models - a simple feed-forward neural net written in NumPy and a CNN written in Tensorflow. One fun feature of this app was online learning: the model continuously improved its predictions based on new inputs (though this did sometimes lead to incorrect predictions).

In 2019 I decided to update the [project](https://github.com/Erlemar/digit-draw-predict): I trained the new neural net in PyTorch and used cv2 to detect separate digits (people often drew multiple digits). More than that, the model had 11 classes - I made a separate class for "junk", as people often drew things for fun: animals, objects, or words.

The first two versions were deployed on Heroku's free plan, but in 2022 these plans were discontinued. I didn't want my project to die because of nostalgia, so I developed a new version and deployed it differently. The current version has an object detection model (yolo3 written from scratch) and 12 classes (digits, junk, and **censored**).
If you want to know what does **censored** means, just try to draw something 😉

Initially, I considered deploying the app on Streamlit Cloud, but its computational limits were too low, so now the model is live on HuggingFace Spaces.

### Links with additional information:

* [Project page on my personal website](https://andlukyane.com/project/drawn-digits-prediction)
* [A dataset with the digits and bounding boxes on Kaggle](https://www.kaggle.com/datasets/artgor/handwritten-digits-and-bounding-boxes)
* [Training code](https://github.com/Erlemar/pytorch_tempest_pet_)
* Blogpost on my personal website
* [Blogpost on medium](https://towardsdatascience.com/the-third-life-of-a-personal-pet-project-for-handwritten-digit-recognition-fd908dc8e7a1)
* Russian blogpost on habrahabr
* [Project code on GitHub](https://github.com/Erlemar/digit-draw-detect)
"""

st.markdown(text, unsafe_allow_html=True)
