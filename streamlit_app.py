import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

# Set page config
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
    layout="wide"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_model():
    """Load the pretrained model"""
    try:
        # First try to load from local file
        if os.path.exists('model.v1.h5'):
            model = tf.keras.models.load_model('model.v1.h5')
            return model
        
        # If not found locally, download from Google Drive
        # Google Drive direct download URL for your model
        model_url = "https://drive.google.com/uc?id=1c5aW3RTb7zqZGqFrG0cVfM6rdloJdTr3"
        
        with st.spinner("Downloading model (first time only)... This may take a few minutes."):
            # Download the model
            import urllib.request
            
            # Create a progress bar for download
            def download_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) / total_size)
                    st.progress(percent / 100)
            
            urllib.request.urlretrieve(model_url, 'model.v1.h5', download_progress)
            model = tf.keras.models.load_model('model.v1.h5')
            st.success("Model downloaded and loaded successfully!")
            return model
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please check if the Google Drive link is correct and publicly accessible.")
        return None

# Cache the CSV loading
@st.cache_data
def load_class_labels():
    """Load class labels from CSV"""
    try:
        df = pd.read_csv('image_dataset.csv')
        # Extract unique classes from the dataset
        classes = sorted(df['label'].unique())
        return classes, df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return [], pd.DataFrame()

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction
    Adjust target_size based on your model's input requirements
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values (common for most models)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image_array, class_names):
    """Make prediction on the preprocessed image"""
    try:
        # Get prediction
        predictions = model.predict(image_array)
        
        # Get predicted class index
        predicted_class_idx = np.argmax(predictions[0])
        
        # Get confidence score
        confidence = predictions[0][predicted_class_idx]
        
        # Get predicted class name
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def main():
    # Title and description
    st.title("🛰️ Satellite Image Classification")
    st.markdown("### Upload a satellite image to classify it using our pretrained model")
    
    # Load model and class labels
    model = load_model()
    class_names, df = load_class_labels()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please check if 'model.v1.h5' exists in the project directory.")
        return
    
    if len(class_names) == 0:
        st.error("❌ Class labels could not be loaded. Please check if 'image_dataset.csv' exists in the project directory.")
        return
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"**Classes Available:** {len(class_names)}")
        
        # Display class names
        st.subheader("🏷️ Class Labels:")
        for i, class_name in enumerate(class_names):
            st.write(f"{i+1}. {class_name}")
        
        # Display dataset info
        if not df.empty:
            st.subheader("📈 Dataset Statistics:")
            class_counts = df['label'].value_counts()
            st.write(class_counts)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Upload a satellite image for classification"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add predict button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying image..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(
                        model, processed_image, class_names
                    )
                    
                    if predicted_class is not None:
                        # Store results in session state for display in col2
                        st.session_state.prediction_results = {
                            'predicted_class': predicted_class,
                            'confidence': confidence,
                            'all_predictions': all_predictions,
                            'class_names': class_names
                        }
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Display results if available
        if 'prediction_results' in st.session_state:
            results = st.session_state.prediction_results
            
            # Main prediction
            st.success(f"**Predicted Class:** {results['predicted_class']}")
            st.info(f"**Confidence:** {results['confidence']:.2%}")
            
            # Progress bar for confidence
            st.progress(float(results['confidence']))
            
            # Show top predictions
            st.subheader("📈 All Class Probabilities")
            
            # Create a DataFrame for better visualization
            prob_df = pd.DataFrame({
                'Class': results['class_names'],
                'Probability': results['all_predictions']
            }).sort_values('Probability', ascending=False)
            
            # Display as bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(prob_df['Class'][::-1], prob_df['Probability'][::-1])
            ax.set_xlabel('Probability')
            ax.set_title('Class Probabilities')
            ax.set_xlim(0, 1)
            
            # Color the highest bar differently
            bars[len(bars)-1].set_color('red')
            
            # Add percentage labels
            for i, (class_name, prob) in enumerate(zip(prob_df['Class'][::-1], prob_df['Probability'][::-1])):
                ax.text(prob + 0.01, i, f'{prob:.1%}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display detailed probabilities table
            st.subheader("📋 Detailed Probabilities")
            prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.4f} ({x:.2%})")
            st.dataframe(prob_df, use_container_width=True)
        
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This app uses a pretrained model for satellite image classification. Results may vary based on image quality and content.")

if __name__ == "__main__":
    main()
