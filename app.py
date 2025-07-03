import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageOps, ImageDraw
import os

# Inject custom CSS for a modern green theme
st.markdown('''
    <style>
    body {
        background-color: #e8f5e9;
    }
    .stApp {
        background-color: #e8f5e9;
    }
    .css-18e3th9 {
        background-color: #e8f5e9 !important;
    }
    .st-bb {
        color: #388e3c !important;
    }
    .st-bb h1, .st-bb h2, .st-bb h3, .st-bb h4, .st-bb h5, .st-bb h6 {
        color: #1b5e20 !important;
    }
    .stButton>button {
        background-color: #43a047;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.5em;
        font-size: 1.1em;
        font-weight: 600;
        margin-top: 1em;
        transition: background 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e7d32;
        color: #fff;
    }
    .stFileUploader label {
        color: #388e3c;
        font-weight: 600;
    }
    .st-cq {
        color: #388e3c !important;
    }
    .stDataFrame, .stTable {
        background-color: #f1f8e9;
    }
    </style>
''', unsafe_allow_html=True)

# Load label encoder classes
LABELS = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Update if your classes differ

# Load the trained model (choose transfer_model.keras for best accuracy)
MODEL_PATH = os.path.join('model', 'transfer_model.keras')
model = load_model(MODEL_PATH)

# Mapping from short code to full name and description
CLASS_INFO = {
    'akiec': ('Actinic keratoses (precancerous)', 'Precancerous skin lesion, may develop into cancer.'),
    'bcc':   ('Basal cell carcinoma', 'A common form of skin cancer.'),
    'bkl':   ('Benign keratosis-like lesions', 'Non-cancerous skin growths.'),
    'df':    ('Dermatofibroma', 'Benign skin nodule.'),
    'mel':   ('Melanoma', 'Serious form of skin cancer.'),
    'nv':    ('Melanocytic nevi (mole)', 'Common mole, usually benign.'),
    'vasc':  ('Vascular lesions', 'Lesions related to blood vessels, usually benign.')
}

st.markdown('<h1 style="color:#1b5e20;text-align:center;font-size:2.8em;font-weight:700;">Skin Cancer Detector</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#388e3c;text-align:center;font-size:1.2em;">Upload a skin lesion image (JPG/PNG) to predict the type of skin cancer.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    pred = model.predict(img_array)
    pred_class = np.argmax(pred, axis=1)[0]
    pred_label = LABELS[pred_class]
    pred_label_full, pred_label_desc = CLASS_INFO[pred_label]
    st.markdown(f'<h2 style="color:#2e7d32;">Prediction: <span style="color:#43a047;">{pred_label_full}</span></h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:1.1em;color:#388e3c;margin-bottom:1em;">{pred_label_desc}</div>', unsafe_allow_html=True)

    # Show class probabilities as ranked percentages with progress bars and full names
    st.markdown('<h4 style="color:#388e3c;">Class probabilities:</h4>', unsafe_allow_html=True)
    probs = pred[0]
    prob_percent = probs * 100
    ranked = sorted(enumerate(prob_percent), key=lambda x: x[1], reverse=True)
    for idx, pct in ranked:
        code = LABELS[idx]
        name, desc = CLASS_INFO[code]
        is_pred = idx == pred_class
        bar_color = '#43a047' if is_pred else '#a5d6a7'
        label_style = 'font-weight:700;color:#2e7d32;' if is_pred else 'font-weight:500;color:#1b5e20;'
        st.markdown(
            f'<div style="margin-bottom:0.2em;"><span style="{label_style}">{name}</span> '
            f'<span style="color:#388e3c;font-weight:600;">({pct:.1f}%)</span></div>'
            f'<div style="font-size:0.9em;color:#388e3c;margin-bottom:0.3em;">{desc}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'''<div style="background:#e0e0e0;border-radius:8px;height:18px;width:100%;margin-bottom:0.7em;">
                <div style="background:{bar_color};width:{pct:.1f}%;height:100%;border-radius:8px;"></div></div>''',
            unsafe_allow_html=True
        )

    # Calculate cancer risk (sum of malignant probabilities)
    malignant_indices = [LABELS.index(x) for x in ['akiec', 'bcc', 'mel']]
    cancer_risk = sum([prob_percent[i] for i in malignant_indices])
    st.markdown(
        f'<h3 style="color:#b71c1c;">Estimated Cancer Risk: <span style="color:#d32f2f;">{cancer_risk:.1f}%</span></h3>',
        unsafe_allow_html=True
    )
    st.caption('Top class highlighted. Probabilities shown as percentages. Cancer risk is the combined probability of malignant classes.')
