import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="🛰️ Satellite Image Classifier",
    page_icon="🌍",
    layout="wide"
)

# For demonstration purposes - mock prediction function
def mock_predict(image_array, class_names):
    """
    Mock prediction function for demonstration
    Replace this with your actual model prediction when TensorFlow is available
    """
    # Generate random predictions for demo
    np.random.seed(42)  # For consistent results
    predictions = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    
    # Get predicted class index
    predicted_class_idx = np.argmax(predictions)
    
    # Get confidence score
    confidence = predictions[predicted_class_idx]
    
    # Get predicted class name
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions

# Cache the CSV loading
@st.cache_data
def load_class_labels():
    """Load class labels from CSV or use default classes"""
    try:
        df = pd.read_csv('image_dataset.csv')
        # Extract unique classes from the dataset
        classes = sorted(df['label'].unique())
        return classes, df
    except Exception as e:
        st.warning(f"CSV not found, using default classes: {e}")
        # Default satellite image classes
        default_classes = [
            'Agricultural Land',
            'Airplane',
            'Baseball Diamond',
            'Beach',
            'Buildings',
            'Chaparral',
            'Dense Residential',
            'Forest',
            'Freeway',
            'Golf Course',
            'Harbor',
            'Intersection',
            'Medium Residential',
            'Mobile Home Park',
            'Overpass',
            'Parking Lot',
            'River',
            'Runway',
            'Sparse Residential',
            'Storage Tanks',
            'Tennis Court'
        ]
        return default_classes, pd.DataFrame()

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the uploaded image for model prediction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(image_array, class_names):
    """Make prediction on the preprocessed image"""
    try:
        # For now, using mock prediction
        # TODO: Replace with actual model prediction when TensorFlow is available
        predicted_class, confidence, predictions = mock_predict(image_array, class_names)
        
        return predicted_class, confidence, predictions
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def main():
    # Title and description
    st.title("🛰️ Satellite Image Classification")
    st.markdown("### Upload a satellite image to classify it using our pretrained model")
    
    # Show warning about TensorFlow
    st.warning("⚠️ **Demo Mode**: TensorFlow is not available. This app is running in demo mode with mock predictions. To use the actual model, deploy on a platform that supports TensorFlow.")
    
    # Load class labels
    class_names, df = load_class_labels()
    
    if len(class_names) == 0:
        st.error("❌ No class labels available.")
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
        
        # Instructions
        st.subheader("📝 Instructions:")
        st.markdown("""
        1. Upload a satellite image
        2. Click 'Classify Image'
        3. View the prediction results
        
        **Note:** Currently in demo mode.
        """)
    
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
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.info(f"**Image Mode:** {image.mode}")
            
            # Add predict button
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying image..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_predictions = predict_image(
                        processed_image, class_names
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
            
            # Display as Streamlit bar chart (no matplotlib needed)
            st.bar_chart(prob_df.set_index('Class')['Probability'])
            
            # Display detailed probabilities table
            st.subheader("📋 Detailed Probabilities")
            prob_df['Probability_Display'] = prob_df['Probability'].apply(lambda x: f"{x:.4f} ({x:.2%})")
            
            # Show top 5 predictions
            st.dataframe(prob_df[['Class', 'Probability_Display']].head(10), use_container_width=True)
        
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This app is currently in demo mode. For production use with actual model predictions, deploy on a platform that supports TensorFlow.")
    
    # Deployment suggestions
    with st.expander("🚀 Deployment Suggestions"):
        st.markdown("""
        **For actual model deployment, consider:**
        
        1. **Hugging Face Spaces** - Better support for ML libraries
        2. **Google Cloud Run** - Scalable container deployment
        3. **AWS EC2** - Full control over environment
        4. **Heroku** - Easy deployment with buildpacks
        5. **Railway** - Modern deployment platform
        
        **To use TensorFlow on Streamlit Cloud:**
        - Use `tensorflow-cpu` instead of `tensorflow`
        - Consider converting model to TensorFlow Lite
        - Optimize model size and memory usage
        """)

if __name__ == "__main__":
    main()
