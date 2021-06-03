import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

try:
    sys.path.remove(str(parent))
except ValueError: 
    pass

import streamlit as st
import validators

from detectron2st import detectron2st
from yolov5st import yolov5st
import os

model_pages = {
    "Detectron2": detectron2st,
    "Yolov5": yolov5st
}

intro = """
This app serves a number of machine learning models using FastAPI and Streamlit.
"""

def draw_main_page():
    st.write(f"""
    # Welcome to Object Detection Toolbox! 👋
    """)

    st.write(intro)

    st.info("""
        :point_left: **To get started, choose a model on the left sidebar.**
    """)


# Draw sidebar
pages = list(model_pages.keys())
pages.insert(0, "Home")

st.sidebar.title(f"Machine Learning Models")
selected_demo = st.sidebar.radio("", pages)

# Draw main page
if selected_demo in model_pages:
    model_pages[selected_demo]()
else:
    draw_main_page()




