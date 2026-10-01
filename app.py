import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

st.set_page_config(page_title="Diabetic Retinopathy Classifier", layout="wide")

# ---------------------------------------------------------
# 1. Pipeline Preprocessing Functions
# ---------------------------------------------------------
def crop_image_from_gray(img, tol=7):
    """Crops uninformative dark outer borders around retinal scans."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)

def process_pipeline_stages(pil_img, sigmaX=10, img_size=224):
    """Executes step-by-step preprocessing and returns intermediate images."""
    raw_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Stage 1: Crop dark borders
    cropped_bgr = crop_image_from_gray(raw_bgr)
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    
    # Stage 2: Spatial resize
    resized_bgr = cv2.resize(cropped_bgr, (img_size, img_size))
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    
    # Stage 3: Ben Graham contrast & edge enhancement
    enhanced_bgr = cv2.addWeighted(resized_bgr, 4, cv2.GaussianBlur(resized_bgr, (0, 0), sigmaX), -4, 128)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(cropped_rgb), Image.fromarray(resized_rgb), Image.fromarray(enhanced_rgb)

# ---------------------------------------------------------
# 2. Model Initialization & Weight Loading
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 5)
    )
    model.load_state_dict(torch.load("resnet50_aptos2019.pth", map_location=device))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = [
    "Stage 0: No DR (Normal)",
    "Stage 1: Mild DR",
    "Stage 2: Moderate DR",
    "Stage 3: Severe DR",
    "Stage 4: Proliferative DR"
]

CLINICAL_INFO = {
    0: ("Normal Retina", "No signs of diabetic retinopathy detected. Routine annual eye examination recommended."),
    1: ("Mild Non-Proliferative", "Microaneurysms detected. Monitor closely; schedule follow-up examination in 6–12 months."),
    2: ("Moderate Non-Proliferative", "Microaneurysms, hemorrhages, or lipid deposits present. Referral to ophthalmologist recommended."),
    3: ("Severe Non-Proliferative", "Extensive intraretinal hemorrhages / microvascular abnormalities. Urgent ophthalmology referral required."),
    4: ("Proliferative DR", "Advanced stage with fragile new blood vessel growth / vitreous hemorrhage. Immediate specialist intervention required.")
}

# ---------------------------------------------------------
# 3. Streamlit Interface
# ---------------------------------------------------------
st.title("👁️ Diabetic Retinopathy Classification Prototype")
st.markdown("Automated diagnostic screening powered by fine-tuned ResNet50 Transfer Learning and OpenCV Ben Graham contrast enhancement.")

uploaded_file = st.file_uploader("Upload Retinal Fundus Scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(raw_img, caption="Raw Input Retinal Scan", use_column_width=True)
        run_btn = st.button("Run Diagnostic Pipeline", type="primary")

    if run_btn:
        # Preprocessing
        cropped_img, resized_img, enhanced_img = process_pipeline_stages(raw_img)
        
        # Inference
        tensor_img = transform(enhanced_img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        predicted_class = int(torch.argmax(probabilities).item())
        confidence_score = float(probabilities[predicted_class]) * 100
        condition_title, clinical_advice = CLINICAL_INFO[predicted_class]
        
        with col2:
            st.subheader("🩺 Diagnostic Summary")
            st.markdown(f"**Predicted Stage:** `{CLASS_NAMES[predicted_class]}`")
            st.markdown(f"**Prediction Confidence:** `{confidence_score:.2f}%`")
            st.markdown(f"**Severity Level:** {condition_title}")
            st.info(f"**Clinical Guidance:** {clinical_advice}")
            
            st.subheader("📊 Stage Confidence Breakdown")
            for i in range(5):
                st.progress(float(probabilities[i]), text=f"{CLASS_NAMES[i]}: {float(probabilities[i])*100:.1f}%")
        
        st.markdown("---")
        st.subheader("🛠️ Step-by-Step Preprocessing Output Visualizer")
        img_col1, img_col2, img_col3 = st.columns(3)
        with img_col1:
            st.image(cropped_img, caption="1. Outer Margin Cropped", use_column_width=True)
        with img_col2:
            st.image(resized_img, caption="2. Resized (224x224)", use_column_width=True)
        with img_col3:
            st.image(enhanced_img, caption="3. Ben Graham Contrast Enhanced", use_column_width=True)
