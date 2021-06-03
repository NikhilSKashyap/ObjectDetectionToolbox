import io
import os
import requests
import json
import streamlit as st
from PIL import Image
from io import BytesIO
import base64

server_url=f"http://0.0.0.0:8008/yolov5"

def yolov5st():
    st.title("YOLOV5")
    st.write(
        """Serving YOLOV5 model using FastAPI and Streamlit."""
    )  
    uploaded_image = st.file_uploader("Upload Image")

    weights = st.selectbox('Select the base model', ('yolov5s.pt','yolov5x.pt'))

    st.write('You selected:', weights)

    threshold_score = st.text_input('Threshold score:')

    if st.button("Detect"):
        if uploaded_image is not None:
            # File details
            file_details = {"FileName": uploaded_image.name,"FileType": uploaded_image.type,"FileSize":uploaded_image.size}
            st.write(file_details)

            with open(os.path.join("Input/",uploaded_image.name),"wb") as f:
                    f.write(uploaded_image.getbuffer())
            
            # File content
            files = {"file": uploaded_image.name , "weights": weights, "threshold_score": threshold_score}
            

            # Post to server
            res = requests.post(server_url, json={"file": uploaded_image.name , "weights": weights, "threshold_score": float(threshold_score)}).json()
            output_image = Image.open('runs/detect/exp/{}'.format(uploaded_image.name))
            st.header("Output Image")
            st.image(output_image, use_column_width=True)

            #download output
            buffered = BytesIO()
            output_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/jpg;base64,{img_str}">Download output</a>'
            
            st.markdown(href, unsafe_allow_html=True)

        else:
            st.write("Upload image!")