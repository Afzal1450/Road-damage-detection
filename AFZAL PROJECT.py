
#Prerequiste
'''
python.exe -m pip install --upgrade pip
pip install cv
pip install numpy
pip install streamlit
pip install PIL
pip install ultralytics
pip install collections
pip install io
pip install tempfile
pip install os
pip install demand
'''
import cv
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import deque
import io
import tempfile
import os
from demand import load_data, show_demand_analysis

def plt_show(image, title=""):
    if len(image.shape) == 3:
        st.image(image, caption=title, use_column_width=True)

def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv.resize(image, dim, interpolation=inter)
    return resized

def road_damage_assessment(uploaded_video):
    import torch

    # Set the device to CPU
    device = torch.device('cpu')

    # Load the YOLO model
    best_model = YOLO('model/best.pt')
    best_model.to(device)

    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    text_position = (40, 80)
    font_color = (255, 255, 255)
    background_color = (0, 0, 255)

    damage_deque = deque(maxlen=20)

    # Save the uploaded video to a temporary file
    temp_video_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(uploaded_video.read())

    cap = cv.VideoCapture(temp_video_path)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('road_damage_assess…:') # Release resources and delete the temporary file
    cap.release()
    out.release()
    cv.destroyAllWindows()
    os.remove(temp_video_path)

def process_uploaded_video(uploaded_video):
    temp_video_file = tempfile.NamedTemporaryFile(delete=False)
    temp_video_file.write(uploaded_video.read())
    temp_video_file_path = temp_video_file.name
    temp_video_file.close()

    road_damage_assessment(temp_video_file_path)

    os.remove(temp_video_file_path)

def main():
    st.set_page_config(page_title="Urban Mobility Solution", page_icon=":car:")
    st.title("Image and Road Damage Assessment")

    # Image Section
    st.markdown("## Image Section")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_image = np.array(image)

        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        resized_image, gray_image = process_image(original_image)

        if gray_image is not None:
            contours = detect_contours(gray_image)

            if contours:
                draw_and_display_contours(resized_image, contours)
                st.success("Pothole Detected!")
            else:
                st.warning("No Pothole Detected!")

            additional_processing_and_display(gray_image)

    # Video Section
    st.markdown("---")  # Separation between image and video sections
    st.markdown("## Video Section")
    uploaded_video = st.file_uploader("Choose a video...", type="mp4")
    
    st.markdown("---") 
    if uploaded_video is not None:
        road_damage_assessment(uploaded_video)

    st.sidebar.markdown("<span style='font-size:28px'>Urban Mobility Solution</span>", unsafe_allow_html=True)
    st.sidebar.markdown("---")  # Separation between image and demand sections
    st.sidebar.markdown("## Demand Prediction")

    # Sidebar menu for demand prediction navigation
    demand_uploaded_file = st.sidebar.file_uploader("Upload CSV file for Demand Prediction", type=["csv"])

    # Load data if file is uploaded
    if demand_uploaded_file:
        df, demand_model = load_data(demand_uploaded_file)
        show_demand_analysis(df)


if _name_ == "_main_":
    main()
