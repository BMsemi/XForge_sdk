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

    with col2:
        st.subheader("🛠 Hardware Stats")
        
        # 3. Use Simulator to analyze a specific layer
        # Here we extract weights from the first layer of YOLO
        layer_weights = sdk["yolo"].get_layer_weights(0)
        
        if layer_weights is not None:
            stats = sdk["sim"].simulate_layer("conv1", layer_weights)
            
            # Display metrics directly from SDK output
            st.metric("PE Utilization", f"{stats['util_pct']:.2f}%")
            st.metric("Total Ops", f"{stats['ops']:,}")
            st.metric("L2 Writes", f"{stats['l2_writes']:,}")
        else:
            st.error("Could not extract weights for simulation.")