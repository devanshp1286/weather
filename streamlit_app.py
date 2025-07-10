import streamlit as st
import numpy as np
import pandas as pd
import os
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# === CONFIG ===
MODEL_ID = "1c5aW3RTb7zqZGqFrG0cVfM6rdloJdTr3"
MODEL_URL = f"https://drive.google.com/uc?id={MODEL_ID}"
MODEL_PATH = "satellite_model.h5"
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
IMAGE_SIZE = (256, 256)

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main > div {
        padding: 2rem 1rem;
    }
    
    /* Custom Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin: 1rem 0 0 0;
        font-weight: 300;
    }
    
    /* Card Styles */
    .card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #2D3748;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Results Section */
    .result-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .prediction-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
        margin-bottom: 1rem;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .confidence-meter {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }
    
    .footer-text {
        color: white;
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Loading Animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f4f6;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Hide Streamlit Elements */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Custom Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: white;
        border-radius: 12px;
        border: 2px dashed #667eea;
        padding: 1.5rem;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# === PAGE CONFIG ===
st.set_page_config(
    page_title="🛰️ Satellite Classifier", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === HEADER ===
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🛰️ Satellite Vision AI</h1>
    <p class="header-subtitle">Advanced satellite image classification powered by deep learning</p>
</div>
""", unsafe_allow_html=True)

# === MODEL LOADING ===
@st.cache_resource
def load_satellite_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔄 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_satellite_model()

# === MAIN LAYOUT ===
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">📤 Upload Satellite Image</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a satellite image (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        help="Upload a satellite image to classify it as Cloudy, Desert, Green Area, or Water"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("🔍 Classify Image"):
            with st.spinner("🤖 Analyzing satellite image..."):
                # Preprocess
                img_array = img_to_array(image) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_array)[0]
                predicted_class = CLASS_NAMES[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                
                # Store in session
                st.session_state.result = {
                    "prediction": predicted_class,
                    "confidence": confidence,
                    "all_probs": prediction
                }
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🔬 Prediction Results</h2>', unsafe_allow_html=True)
    
    if "result" in st.session_state:
        result = st.session_state.result
        
        # Prediction Badge
        st.markdown(f'<div class="prediction-badge">🎯 {result["prediction"]}</div>', unsafe_allow_html=True)
        
        # Confidence Meter
        st.markdown('<div class="confidence-meter">', unsafe_allow_html=True)
        st.markdown(f"**Confidence Level:** {result['confidence'] * 100:.2f}%")
        st.progress(result['confidence'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Enhanced Probability Chart
        st.markdown("### 📊 Class Probabilities")
        df = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": result['all_probs']
        }).sort_values(by="Probability", ascending=False)
        
        # Create a beautiful bar chart with Plotly
        fig = px.bar(
            df, 
            x="Class", 
            y="Probability", 
            color="Probability",
            color_continuous_scale="viridis",
            title="Prediction Confidence by Class"
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12),
            title_font=dict(size=16, color="#2D3748"),
            xaxis_title="Classification Classes",
            yaxis_title="Probability Score",
            showlegend=False,
            height=400
        )
        
        fig.update_traces(
            marker_line_color='rgba(102, 126, 234, 0.8)',
            marker_line_width=2,
            opacity=0.8
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Results Table
        df["Confidence (%)"] = (df["Probability"] * 100).round(2)
        st.markdown("### 📈 Detailed Results")
        st.dataframe(
            df[["Class", "Confidence (%)"]].reset_index(drop=True), 
            use_container_width=True,
            hide_index=True
        )
        
    else:
        st.markdown("""
        <div class="result-card">
            <h3 style="color: #667eea; margin-bottom: 1rem;">🚀 Ready to Analyze</h3>
            <p style="color: #64748b; margin: 0;">Upload a satellite image and click 'Classify Image' to see AI-powered predictions with confidence scores and detailed probability breakdown.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# === INFORMATION SECTION ===
st.markdown("""
<div class="card">
    <h2 class="card-title">ℹ️ About This Classifier</h2>
    <p style="color: #64748b; line-height: 1.6; margin: 0;">
        This AI model can identify four main types of satellite imagery: <strong>Cloudy</strong> areas, 
        <strong>Desert</strong> regions, <strong>Green Areas</strong> (vegetation), and <strong>Water</strong> bodies. 
        The model uses deep learning to analyze pixel patterns and provide accurate classifications with confidence scores.
    </p>
</div>
""", unsafe_allow_html=True)

# === FOOTER ===
st.markdown("""
<div class="footer">
    <p class="footer-text">Built with ❤️ using Streamlit, TensorFlow & Modern UI Design</p>
</div>
""", unsafe_allow_html=True)
