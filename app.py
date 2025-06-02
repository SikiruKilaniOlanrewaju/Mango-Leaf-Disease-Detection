import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Load the trained model
MODEL_PATH = 'best_model.pth'
CLASSES = ['Anthracnose', 'Bacterial_Canker', 'Die_Back', 'Gall_Mid', 'healthy']  # Update if your classes are different

def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return CLASSES[predicted.item()]

# --- Enhanced Streamlit App for Mango Leaf Disease Detection ---

st.set_page_config(
    page_title="Anthracnose in Mango Disease Detection",
    page_icon="🍃",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background: #228B22;
        border-radius: 8px;
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .stFileUploader>div>div>button {
        background: #228B22;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title('🍃 Anthracnose Mango Leaf Disease Detection')
st.markdown('''
Upload a mango leaf image to detect **Anthracnose** or other diseases using a deep learning model.

- You can check here if a mango leaf has Anthracnose or not
- Model: ResNet18 (PyTorch)
''')

with st.sidebar:
    st.header("About")
    st.write("This app uses a deep learning model to classify mango leaf diseases. Upload a clear image of a mango leaf for best results.")
    st.markdown("""
    **Developed by:** Your Name  
    **Date:** 2025-05-27
    """)

uploaded_file = st.file_uploader('Choose a mango leaf image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Analyzing...'):
        model = load_model()
        prediction = predict(image, model)
    st.success(f'Prediction: **{prediction}**')
    st.balloons()
else:
    st.info('Please upload a mango leaf image to get started.')

st.markdown("---")
st.caption("© 2025 Mango Leaf Disease Detection | Using Deep Learning")
