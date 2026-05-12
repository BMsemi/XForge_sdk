import streamlit as st
import numpy as np
from XForge import NeuromorphicSimulator, ImageProcessor, YOLOWrapper

# Page Config
st.set_page_config(page_title="NeuroSim Dashboard", layout="wide")

# 1. Initialize SDK Components
# These replace the manual logic previously scattered in app.py and user.py
@st.cache_resource
def init_sdk():
    return {
        "sim": NeuromorphicSimulator(num_pes=64),
        "processor": ImageProcessor(),
        "yolo": YOLOWrapper('yolov8n.pt')
    }

sdk = init_sdk()

st.title("🎯 YOLO26N: Neuromorphic Analysis SDK")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    # 2. Use ImageProcessor for standardized loading
    img_array, channels = sdk["processor"].load_rgb_image(uploaded_file)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Object Detection")
        results = sdk["yolo"].predict(img_array)
        st.image(results[0].plot(), caption="Detection Results")