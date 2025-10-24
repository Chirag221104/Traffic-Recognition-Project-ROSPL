import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

class BasicImageEnhancer:
    
    def __init__(self):
        self.denoising_autoencoder = None
    
    def apply_clahe(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=0.03, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = image
        return enhanced
    
    def apply_edge_enhancement(self, image):
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        return enhanced
    
    def apply_denoising(self, image):
        if self.denoising_autoencoder is not None:
            try:
                image_input = np.expand_dims(image.astype('float32') / 255.0, axis=0)
                denoised = self.denoising_autoencoder.predict(image_input, verbose=0)
                return (denoised[0] * 255).astype(np.uint8)
            except:
                pass
        # Fallback to simple denoising
        return cv2.GaussianBlur(image, (5, 5), 0)
    
    def apply_combined(self, image):
        enhanced = self.apply_clahe(image)
        enhanced = self.apply_edge_enhancement(enhanced)
        return enhanced

def run():
    # Configure page
    st.set_page_config(page_title="Traffic Sign Recognition", layout="wide")
    
    st.title("Traffic Sign Recognition App")
    st.write("Upload a traffic sign image to see enhancement comparison and predictions")
    
    # Model path
    model_path = st.text_input("Model Path:", value="../models/final_model.h5")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.info("Make sure your model file exists at the specified path")
        return
    
    st.write(f"Loading model from {model_path}...")
    
    try:
        with st.spinner('Loading model...'):
            model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Initialize enhancer
    enhancer = BasicImageEnhancer()
    
    # Try loading autoencoder
    autoencoder_path = "./outputs/denoising_autoencoder.h5"
    if os.path.exists(autoencoder_path):
        try:
            with st.spinner('Loading autoencoder...'):
                enhancer.denoising_autoencoder = tf.keras.models.load_model(autoencoder_path)
            st.info("Autoencoder loaded for denoising")
        except:
            st.warning("Autoencoder found but couldn't load")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a traffic sign image...", 
                                   type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Load and resize image
        image = Image.open(uploaded_file).convert('RGB')
        image_array = np.array(image)
        
        # Resize to model input size (30x30 as per your original code)
        resized_image = cv2.resize(image_array, (30, 30))
        
        st.subheader("Uploaded Image")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Original', use_container_width=True)
        with col2:
            st.image(resized_image, caption='Resized (30x30)', use_container_width=True)
        
        # Apply different enhancements
        st.subheader("Enhancement Comparison")
        
        # Generate all enhanced versions
        with st.spinner('Applying enhancements...'):
            original = resized_image
            clahe_enhanced = enhancer.apply_clahe(original)
            edge_enhanced = enhancer.apply_edge_enhancement(original)
            denoised = enhancer.apply_denoising(original)
            combined = enhancer.apply_combined(original)
        
        enhanced_images = {
            'Original': original,
            'CLAHE': clahe_enhanced,
            'Edge Enhanced': edge_enhanced,
            'Denoised': denoised,
            'Combined': combined
        }
        
        # Display images in columns
        cols = st.columns(5)
        predictions = {}
        
        with st.spinner('Making predictions...'):
            for i, (method, img) in enumerate(enhanced_images.items()):
                with cols[i]:
                    st.image(img, caption=method, use_container_width=True)
                    
                    # Make prediction
                    img_normalized = img.astype('float32') / 255.0
                    img_batch = np.expand_dims(img_normalized, axis=0)
                    
                    prediction = model.predict(img_batch, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    predictions[method] = {
                        'class': predicted_class,
                        'confidence': confidence
                    }
                    
                    # Display prediction below image
                    st.markdown(f"**Class: {predicted_class}**")
                    st.markdown(f"**Conf: {confidence:.1f}%**")
        
        # Summary table
        st.subheader("Prediction Summary")
        
        summary_data = []
        for method, result in predictions.items():
            summary_data.append({
                'Enhancement Method': method,
                'Predicted Class': result['class'],
                'Confidence (%)': f"{result['confidence']:.2f}"
            })
        
        st.table(summary_data)
        
        # Best method
        best_method = max(predictions.keys(), key=lambda x: predictions[x]['confidence'])
        best_result = predictions[best_method]
        
        st.success(f"**Best Enhancement: {best_method}**  \n"
                  f"Predicted Class: **{best_result['class']}**  \n" 
                  f"Confidence: **{best_result['confidence']:.2f}%**")
        
        #
        # Show improvement over original
        original_conf = predictions['Original']['confidence']
        best_conf = best_result['confidence']
        if best_conf > original_conf:
            improvement = best_conf - original_conf
            st.info(f"Improvement over original: **+{improvement:.2f}%**")
        elif best_method != 'Original':
            st.info(f"Original image already performs best")
        
        # Additional enhancement controls
        with st.expander("🔧 Advanced Enhancement Controls"):
            st.subheader("Custom Enhancement Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                clahe_clip = st.slider("CLAHE Clip Limit", 0.01, 0.10, 0.03, 0.01)
                edge_strength = st.slider("Edge Enhancement Strength", 0.5, 3.0, 1.5, 0.1)
            
            with col2:
                blur_kernel = st.selectbox("Denoising Kernel Size", [3, 5, 7, 9], index=1)
                st.info("Adjust parameters and re-upload image to see changes")

if __name__ == "__main__":
    run()