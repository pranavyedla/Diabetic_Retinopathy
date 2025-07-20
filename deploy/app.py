import streamlit as st
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2
import os

st.set_page_config(
    page_title="DR Classifier",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    .css-18ni7ap {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    body {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-bottom: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model function
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'full_model.pth')
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            return model
        else:
            st.error("Model file 'full_model.pth' not found in the deploy folder. Please ensure the model file is present.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_model()
names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# Check if model loaded successfully
if model is None:
    st.stop()

# Image preprocessing
def crop_image_from_gray(img, tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    if mask.any():
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
    return img

# Transformations
val_transforms = A.Compose([
    A.Resize(height=728, width=728),
    A.Normalize(mean=(0.3199, 0.2240, 0.1609),
                std=(0.3020, 0.2183, 0.1741),
                max_pixel_value=255.0),
    ToTensorV2(),
])

# Prediction function
def make_prediction(model, img):
    x = val_transforms(image=img)["image"].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(x)
        pred[pred < 0.75] = 0
        pred[(pred >= 0.75) & (pred < 1.5)] = 1
        pred[(pred >= 1.5) & (pred < 2.4)] = 2
        pred[(pred >= 2.4) & (pred < 3.4)] = 3
        pred[pred >= 3.4] = 4
        pred = pred.long().squeeze(1)
    return pred.cpu().numpy()[0]

# Modern UI Styling
st.markdown("""
    <style>
        h1 {color: #2c3e50; text-align: center; font-size: 40px; margin-bottom: 10px;}
        h4 {text-align: center; color: #7b1fa2;}
        .stButton>button {
            background: linear-gradient(to right, #43cea2, #185a9d);
            color: white;
            padding: 10px 25px;
            font-size: 16px;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(to right, #185a9d, #43cea2);
        }
        .prediction {
            font-size: 26px;
            font-weight: bold;
            color: #d32f2f;
            text-align: center;
            margin-top: 20px;
        }
        .team-card {
            background: #ffffff;
            padding: 15px;
            margin: 10px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
        .team-name {
            font-weight: bold;
            color: #2e7d32;
        }
        .role {
            font-style: italic;
            color: #5e35b1;
        }
        footer {
            font-size: 15px;
            color: #6c757d;
            text-align: center;
            margin-top: 50px;
            border-top: 1px solid #ccc;
            padding-top: 10px;
        }
        /* Responsive tweaks for mobile */
        @media (max-width: 600px) {
            h1 { font-size: 1.5rem !important; }
            h3, h4 { font-size: 1.1rem !important; }
            .stButton>button { font-size: 14px !important; padding: 8px 16px !important; }
            .prediction { font-size: 18px !important; }
            .team-card { padding: 8px !important; }
            img { max-width: 100% !important; height: auto !important; }
            .block-container { padding: 0.5rem !important; }
        }
        /* Responsive team names row/column */
        .team-names {
            display: flex;
            flex-direction: row;
            gap: 1.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        @media (max-width: 600px) {
            .team-names {
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;font-size: 2.5rem;color:cyan'> Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

# Centered upload section
st.markdown('<h3 style="text-align:center; margin-bottom: 0.5rem;font-size: 1.4rem;">Upload an eye image to analyze for diabetic retinopathy severity.</h3>', unsafe_allow_html=True)

# Centered, smaller uploader
col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded_image = st.file_uploader("Upload Eye Image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'], key="centered-uploader")

if uploaded_image:
    img = Image.open(uploaded_image).convert('RGB')
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(img.resize((512, 512)), caption="Uploaded Eye Image")
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze Image")
    processed_img = crop_image_from_gray(np.array(img))
    if analyze_clicked:
        with st.spinner("Analyzing..."):
            result = make_prediction(model, processed_img)
            st.markdown("---")
            st.markdown(f'<div class="prediction"> Result: <span style="color:#1976d2">{names[result]}</span></div>', unsafe_allow_html=True)

# Styled Footer with team members
st.markdown("---")
# Footer
st.markdown("""
    <div style="
        width: 100%;
        margin-top: 30px;
        padding-top: 20px;
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        border-top: 1px #e0e0e0;
    ">
        © 2025 Pranav
    </div>
""", unsafe_allow_html=True)
