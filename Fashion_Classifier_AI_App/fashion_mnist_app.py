import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Fashion Classifier",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache the model in memory
@st.cache_resource
def load_fashion_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(script_dir, "fashion_cnn_model.keras"),
            "fashion_cnn_model.keras",
            "Fashion_Classifier_AI_App/fashion_cnn_model.keras",
            "../fashion_cnn_model.keras",
        ]
        
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    model = load_model(path)
                    return model
            except Exception as e:
                continue
        
        st.error("⚠️ Model file not found!")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

# Class names
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Emojis for each class
class_emojis = ['👕', '👖', '🧥', '👗', '🧥', '👡', '👔', '👟', '👜', '👢']

# Image processing function
def preprocess_image(image):
    img_gray = image.convert('L')
    img_resized = img_gray.resize((28, 28))
    img_array = np.array(img_resized)
    
    if img_array.mean() > 127:
        img_array = 255 - img_array
    
    img_normalized = img_array / 255.0
    img_final = img_normalized.reshape(1, 28, 28, 1)
    
    return img_final, img_array

# Prediction function
def predict_image(model, img_processed):
    predictions = model.predict(img_processed, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class] * 100
    
    return predicted_class, confidence, predictions[0]

# Modern, clean styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        padding: 2rem 1rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 2rem;
    }
    .main-title h1 {
        color: #2c3e50;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    .main-title p {
        color: #6c757d;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Result card - special styling */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        text-align: center;
        margin: 1rem 0;
    }
    .result-card h2 {
        font-size: 3rem;
        margin: 1rem 0;
        font-weight: 700;
    }
    .result-card .emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Confidence bar */
    .confidence-container {
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 0.5rem;
        margin-top: 1rem;
    }
    .confidence-bar {
        height: 40px;
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        transition: width 0.5s ease;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 2px dashed #cbd5e0;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .upload-section:hover {
        border-color: #667eea;
    }
    
    /* Streamlit elements customization */
    .stFileUploader > div > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: transform 0.2s ease;
    }
    .stFileUploader > div > button:hover {
        transform: scale(1.05);
    }
    
    /* Info boxes */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-box h4 {
        color: #1976d2;
        margin-top: 0;
    }
    .info-box ul {
        margin-bottom: 0;
        color: #424242;
    }
    
    /* Developer card */
    .developer-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: center;
        color: white;
        margin-top: 3rem;
    }
    .developer-name {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .developer-title {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    .social-link {
        display: inline-block;
        margin: 0.5rem;
        padding: 0.5rem 1.5rem;
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        color: white;
        text-decoration: none;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .social-link:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title h1 {
            font-size: 1.8rem;
        }
        .result-card h2 {
            font-size: 2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
    <div class="main-title">
        <h1>👗 Fashion Classifier AI</h1>
        <p>Identify clothing items instantly with AI</p>
    </div>
""", unsafe_allow_html=True)

# Load model
model = load_fashion_model()

if model is not None:
    # Input section with tabs for better UX
    tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Take Photo"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['png', 'jpg', 'jpeg'],
            help="Best results with clear images on white background",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        camera_image = st.camera_input(
            "Take a photo",
            help="Position the clothing item clearly in the frame",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process image
    image_to_process = None
    
    if uploaded_file is not None:
        image_to_process = Image.open(uploaded_file)
    elif camera_image is not None:
        image_to_process = Image.open(camera_image)
    
    if image_to_process is not None:
        st.markdown("---")
        
        # Process and predict
        img_processed, img_display = preprocess_image(image_to_process)
        predicted_class, confidence, all_predictions = predict_image(model, img_processed)
        
        # Results section
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📷 Your Image")
            st.image(image_to_process, use_container_width=True)
            
            # Show processed image in small
            st.caption("Processed for AI (28x28 grayscale)")
            st.image(img_display, width=150)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Main result with gradient background
            st.markdown(f"""
                <div class="result-card">
                    <div class="emoji">{class_emojis[predicted_class]}</div>
                    <h2>{class_names[predicted_class]}</h2>
                    <div class="confidence-container">
                        <div class="confidence-bar" style="width: {confidence}%">
                            {confidence:.1f}% Confidence
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # All predictions chart
        st.markdown("---")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Detailed Predictions")
        
        # Create more colorful chart
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')
        
        y_pos = np.arange(len(class_names))
        colors = ['#667eea' if i == predicted_class else '#cbd5e0' for i in range(len(class_names))]
        bars = ax.barh(y_pos, all_predictions * 100, color=colors, height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{class_emojis[i]} {class_names[i]}" for i in range(len(class_names))])
        ax.set_xlabel('Confidence (%)', fontsize=12, color='#2c3e50', fontweight='600')
        ax.set_title('Classification Confidence for All Categories', fontsize=14, color='#2c3e50', fontweight='600', pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e0')
        ax.spines['bottom'].set_color('#cbd5e0')
        ax.tick_params(colors='#6c757d')
        ax.grid(axis='x', alpha=0.2, linestyle='--')
        
        # Add percentage labels
        for i, v in enumerate(all_predictions * 100):
            color = 'white' if i == predicted_class else '#6c757d'
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='600', color=color)
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="info-box">
            <h4>⚙️ Setup Required</h4>
            <ol>
                <li>Train your model and save it as <code>fashion_cnn_model.keras</code></li>
                <li>Place the model file in the <code>Fashion_Classifier_AI_App</code> folder</li>
                <li>Refresh this page</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Tips section
st.markdown("---")
st.markdown("""
    <div class="info-box">
        <h4>💡 Tips for Best Results</h4>
        <ul>
            <li>Use clear, well-lit images of clothing items</li>
            <li>White or light backgrounds work best</li>
            <li>Ensure the item fills most of the frame</li>
            <li>Avoid cluttered backgrounds</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Developer card
st.markdown("""
    <div class="developer-card">
        <div class="developer-name">Marwan Al-Masrrat</div>
        <div class="developer-title">🤖 AI & Machine Learning Enthusiast</div>
        <div>
            <a href="https://www.linkedin.com/in/marwan-al-masrat" target="_blank" class="social-link">
                🔗 LinkedIn
            </a>
        </div>
        <p style="margin-top: 1rem; opacity: 0.8; font-size: 0.9rem;">
            Building intelligent solutions with deep learning
        </p>
    </div>
""", unsafe_allow_html=True)
