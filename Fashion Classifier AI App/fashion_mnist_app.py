import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="wide"
)

# Cache the model in memory
@st.cache_resource
def load_fashion_model():
    try:
        model = load_model("fashion_cnn_model.keras")
        return model
    except:
        st.error("⚠️ Model file not found! Make sure 'fashion_cnn_model.keras' is in the same folder.")
        return None

# Class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Image processing function
def preprocess_image(image):
    # Convert to grayscale
    img_gray = image.convert('L')
    
    # Resize to 28x28
    img_resized = img_gray.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img_resized)
    
    # Invert colors if background is light
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    # Normalize (0-1)
    img_normalized = img_array / 255.0
    
    # Reshape for model
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    return img_final, img_array

# Prediction function
def predict_image(model, img_processed):
    predictions = model.predict(img_processed, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence, predictions[0]

# Application interface
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .title-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .confidence-bar {
        height: 30px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        text-align: center;
        color: white;
        line-height: 30px;
        font-weight: bold;
    }
    .developer-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        text-align: center;
        color: white;
        margin-top: 2rem;
    }
    .developer-name {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .developer-title {
        font-size: 18px;
        margin-bottom: 15px;
        opacity: 0.9;
    }
    .tech-icons {
        font-size: 16px;
        margin: 15px 0;
    }
    .linkedin-link {
        display: inline-block;
        margin-left: 15px;
        color: white;
        text-decoration: none;
        padding: 8px 15px;
        background: rgba(255,255,255,0.2);
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .linkedin-link:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="title-container">
        <h1>👗 Fashion Classifier AI 🎽</h1>
        <p style='font-size: 18px; color: #666;'>Upload an image or take a photo to classify the clothing item</p>
    </div>
""", unsafe_allow_html=True)

# تحميل النموذج
model = load_fashion_model()

if model is not None:
    # Input method selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image of a clothing item on a white background for best results"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📸 Take Photo")
        camera_image = st.camera_input(
            "Capture photo",
            help="Take a live photo of the clothing item"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process image
    image_to_process = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif camera_image is not None:
        image_to_process = Image.open(camera_image)
    
    if image_to_process is not None:
        # Display results
        st.markdown("---")
        
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("🖼️ Original Image")
            st.image(image_to_process, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Process and predict
        img_processed, img_display = preprocess_image(image_to_process)
        predicted_class, confidence, all_predictions = predict_image(model, img_processed)
        
        with col_result:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.subheader("🔍 Result")
            
            # Display processed image
            st.image(img_display, caption="Processed Image (28x28)", width=200)
            
            # Main result
            st.markdown(f"""
                <div style='text-align: center; padding: 1rem;'>
                    <h2 style='color: #667eea; margin: 0;'>{class_names[predicted_class]}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown("### Confidence Level")
            st.markdown(f"""
                <div class="confidence-bar" style="width: {confidence}%">
                    {confidence:.2f}%
                </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Display all probabilities
        st.markdown("---")
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.subheader("📊 All Predictions")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(class_names))
        bars = ax.barh(y_pos, all_predictions * 100)
        
        # Color the highest bar
        bars[predicted_class].set_color('#667eea')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('Percentage (%)', fontsize=12)
        ax.set_title('Probability Distribution for All Classes', fontsize=14, pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(all_predictions * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')
        
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
        <div class='result-box'>
            <h3>⚠️ Usage Instructions:</h3>
            <ol>
                <li>Make sure to run the code in the Notebook first to create the model</li>
                <li>Save the model using: <code>model.save("fashion_cnn_model.keras")</code></li>
                <li>Place the <code>fashion_cnn_model.keras</code> file in the same folder as the app</li>
                <li>Run the app using: <code>streamlit run app.py</code></li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Additional information
st.markdown("---")
st.markdown("""
    <div class='result-box'>
        <h4>💡 Tips for Best Results:</h4>
        <ul>
            <li>Use clear images of clothing items</li>
            <li>Prefer white or light backgrounds</li>
            <li>Make sure the clothing item is clearly visible in the image</li>
            <li>The model is trained on 28x28 pixel grayscale images</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Developer Card
st.markdown("---")
st.markdown("""
    <div class="developer-card">
        <div class="developer-name">By Marwan Al-Masrrat</div>
        <div class="developer-title">💻 AI Enthusiast</div>
        <div class="tech-icons">🐍 Python | 🤖 AI/ML
            <a href="https://www.linkedin.com/in/marwan-al-masrat" target="_blank" class="linkedin-link">
                🔗 LinkedIn
            </a>
        </div>
        <p style="margin: 15px 0 0 0; font-size: 14px; color: white; opacity: 0.8;">
            Building intelligent solutions with machine learning
        </p>
    </div>
""", unsafe_allow_html=True)