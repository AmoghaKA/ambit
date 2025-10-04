# app.py
"""
Streamlit app:
- Upload biopsy report image -> OCR -> auto-fill 10 mean_* features -> predict using model_pipeline_10.pkl
- Show SHAP explanation
- Optional Mammogram demo (ResNet50 placeholder)
"""

import io
import os
import re
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# OCR libs
from PIL import Image
import pytesseract
import easyocr
import cv2

# ML / SHAP / plotting
import shap
import matplotlib.pyplot as plt

# For mammogram demo
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

st.set_page_config(page_title="OncoScan AI (OCR + ML Demo)", layout="wide")
st.title("OncoScan AI — OCR → Tabular Model → Explainability")

# Provide tesseract path safety (adjust if necessary)
# If you added Tesseract to PATH, this isn't required. Keep it for reliability.
DEFAULT_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(DEFAULT_TESSERACT):
    pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT

# Load the 10-feature model
MODEL_FILENAME = "model_pipeline_10.pkl"
if not os.path.exists(MODEL_FILENAME):
    st.error(f"Model file not found: {MODEL_FILENAME}. Run retrain_model.py first.")
    st.stop()

model = joblib.load(MODEL_FILENAME)
# expected feature order (space-separated names, matching sklearn dataset)
FEATURES_10 = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
]

# easyocr reader (lazily initialize)
reader = None
def get_easyocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

# --- Helper functions ---

def preprocess_image_for_ocr(pil_img: Image.Image) -> np.ndarray:
    # convert to grayscale numpy array and apply simple denoising & thresholding
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # optional denoise and threshold
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def ocr_with_pytesseract(img_arr: np.ndarray) -> str:
    # pytesseract expects PIL or numpy array
    text = pytesseract.image_to_string(img_arr)
    return text

def ocr_with_easyocr(pil_img: Image.Image) -> str:
    rdr = get_easyocr_reader()
    results = rdr.readtext(np.array(pil_img))
    # join fragments into lines
    lines = []
    for bbox, text, prob in results:
        lines.append(text)
    return "\n".join(lines)

# robust numeric extractor
NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")

def parse_ocr_text_to_features(ocr_text: str) -> Tuple[Dict[str, float], List[str]]:
    """
    Attempts to parse label:value pairs from OCR text.
    Returns (feature_dict, notes).
    - If labels exist (like 'mean radius: 12.32'), map directly.
    - If only numbers and count == 10, map in order to FEATURES_10.
    """
    notes = []
    feature_dict = {}

    # Normalize text: replace underscores with spaces, lower-case
    txt = ocr_text.replace("_", " ").lower()

    # Try to find label:value lines first
    # pattern: label (letters/spaces) : value
    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    for line in lines:
        # common separators: ":", "-", tab, multiple spaces
        if ":" in line or "-" in line:
            # try split on ':'
            parts = re.split(r":|-|\t", line, maxsplit=1)
            if len(parts) >= 2:
                label = parts[0].strip()
                value_part = parts[1]
                num_match = NUM_RE.search(value_part)
                if num_match:
                    val = float(num_match.group())
                    # normalize label to match FEATURES_10 possible formats
                    label_norm = label.replace(" ", " ").strip()
                    feature_dict[label_norm] = val
    # Try fuzzy matching of labels to real feature names (basic approach)
    # If we found some labels, map them to closest FEATURES_10 by simple substring matching
    if feature_dict:
        mapped = {}
        for raw_label, val in feature_dict.items():
            # direct match
            best = None
            for feat in FEATURES_10:
                if raw_label in feat or feat in raw_label or raw_label.replace(" ", "") in feat.replace(" ", ""):
                    best = feat
                    break
            if best is None:
                # try loosened match: check if any key tokens overlap
                tokens = raw_label.split()
                for feat in FEATURES_10:
                    if any(tok in feat for tok in tokens):
                        best = feat
                        break
            if best:
                mapped[best] = val
            else:
                notes.append(f"Unmapped label '{raw_label}'")
        return mapped, notes

    # If no label:value pairs found, extract all numbers
    all_nums = NUM_RE.findall(txt)
    if len(all_nums) == 0:
        notes.append("No numbers found in OCR text.")
        return {}, notes

    # If exactly 10 numbers, map by order to FEATURES_10
    if len(all_nums) == 10:
        vals = [float(x) for x in all_nums]
        mapped = {feat: v for feat, v in zip(FEATURES_10, vals)}
        notes.append("Detected 10 numeric values — mapped by order to mean_* features.")
        return mapped, notes

    # If there are more numbers, attempt to find the first 10 reasonable floats (heuristic)
    floats = [float(x) for x in all_nums]
    if len(floats) >= 10:
        mapped = {feat: floats[i] for i, feat in enumerate(FEATURES_10)}
        notes.append(f"Detected {len(floats)} numbers — using the first 10 by order.")
        return mapped, notes

    # fallback: if fewer than 10 numbers, map what we have and leave the rest NaN
    mapped = {}
    for i, val in enumerate(all_nums):
        if i < len(FEATURES_10):
            mapped[FEATURES_10[i]] = float(val)
    notes.append(f"Only {len(all_nums)} numeric values found; remaining fields will be left blank.")
    return mapped, notes

def make_input_df(mapped_features: Dict[str, float]) -> pd.DataFrame:
    # prepare full DataFrame with all 10 features, filling missing with the column mean (safe) or np.nan
    # Here we'll fill missing with dataset mean for safety if available; else with 0
    # For simplicity we will fill with the column mean computed from the training dataset via model if possible
    defaults = {}
    # attempt to infer means from training scaler? easier: use rough defaults or zeros
    for feat in FEATURES_10:
        if feat in mapped_features:
            defaults[feat] = mapped_features[feat]
        else:
            defaults[feat] = np.nan  # leave NaN; pipeline scaler will accept it if sklearn >=1.0; else we fill with 0
    df = pd.DataFrame([defaults])
    # If any NaN - replace with column means from training data if available (we can compute from model)
    # Attempt to recover mean from training data if pipeline has attributes (not always available). Fallback to 0.
    if df.isna().any().any():
        try:
            # Try: if pipeline stored training means in scaler, use them (StandardScaler keeps mean_ attribute)
            scaler = model.named_steps.get("scaler", None)
            if scaler is not None and hasattr(scaler, "mean_"):
                # scaler.mean_ corresponds to numeric features *in the fitted order*
                # but scaler was fit on dataframe columns in the same order FEATURES_10
                for idx, feat in enumerate(FEATURES_10):
                    if pd.isna(df.loc[0, feat]):
                        df.loc[0, feat] = scaler.mean_[idx]
            else:
                # fallback: fill NaN with 0 (less ideal but safe)
                df = df.fillna(0.0)
        except Exception:
            df = df.fillna(0.0)
    return df

# --- Streamlit UI ---
st.sidebar.title("Controls")
st.sidebar.write("Upload a biopsy report image or try the demo sample.")
t1, t2 = st.tabs(["Report OCR → Predict", "Mammogram Image Demo (optional)"])

with t1:
    st.header("Upload biopsy report image (image or screenshot)")
    uploaded = st.file_uploader("Upload image (png/jpg) with report values", type=["png", "jpg", "jpeg"])
    sample_button = st.button("Use sample generated report image (demo)")

    if sample_button:
        # load the sample image provided alongside this code (if you saved it) - otherwise we can skip
        try:
            sample_path = "A_scanned_medical_report_of_a_UK_biopsy_results_ti.png"
            if os.path.exists(sample_path):
                uploaded = open(sample_path, "rb")
                uploaded = io.BytesIO(uploaded.read())
            else:
                st.warning("Sample image not found in project folder.")
        except Exception:
            st.warning("Sample load failed.")

    if uploaded:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded report", use_column_width=True)

            # OCR preprocess
            img_for_ocr = preprocess_image_for_ocr(pil_img)

            # try pytesseract first
            ocr_text = ""
            try:
                ocr_text = ocr_with_pytesseract(img_for_ocr)
                if len(ocr_text.strip()) < 5:
                    raise RuntimeError("pytesseract returned empty or short text.")
            except Exception as e:
                # fallback to easyocr
                st.info("pytesseract had trouble — falling back to EasyOCR.")
                ocr_text = ocr_with_easyocr(pil_img)

            st.markdown("**Raw OCR output (first 500 chars):**")
            st.code(ocr_text[:500])

            # parse
            mapped, notes = parse_ocr_text_to_features(ocr_text)
            if notes:
                for n in notes:
                    st.caption(n)

            if not mapped:
                st.error("Could not extract any mapped features. Try a clearer image or a labeled report.")
            else:
                # build input DataFrame with all 10 features
                input_df = make_input_df(mapped)
                st.subheader("Auto-filled values (model input)")
                st.dataframe(input_df.T.rename(columns={0: "value"}))

                # predict
                pred = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0][int(pred)]
                label = "Malignant" if pred == 1 else "Benign"
                color = "red" if pred == 1 else "green"

                st.markdown(f"### 🔎 Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                st.write(f"Confidence (class probability): {pred_proba:.3f}")

                # SHAP explanation
                st.subheader("🧠 Why the model made this decision (SHAP)")
                try:
                    clf = model.named_steps['clf']
                    explainer = shap.Explainer(clf)
                    # shap wants DataFrame with column names matching training features (FEATURES_10)
                    shap_vals = explainer(input_df)
                    # waterfall plot for single prediction
                    fig_w = plt.figure(figsize=(8, 5))
                    shap.waterfall_plot(shap_vals[0], show=False)
                    st.pyplot(fig_w)
                except Exception as e:
                    st.error(f"SHAP plot failed: {e}")

        except Exception as e:
            st.error(f"Failed processing image: {e}")

with t2:
    st.header("Mammogram / Ultrasound Image Demo (optional)")
    st.write("This is a demo area — a pretrained ResNet50 (ImageNet) is used as a placeholder.")
    uploaded_mammo = st.file_uploader("Upload mammogram (jpg/png)", type=["png", "jpg", "jpeg"], key="mammo")
    if uploaded_mammo:
        try:
            img = Image.open(uploaded_mammo).convert("RGB")
            st.image(img, caption="Uploaded image", use_column_width=True)
            # prepare for ResNet50
            img_resized = img.resize((224, 224))
            arr = keras_image.img_to_array(img_resized)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)

            # lazy load ResNet50
            if "resnet_model" not in st.session_state:
                st.session_state["resnet_model"] = ResNet50(weights="imagenet")

            preds = st.session_state["resnet_model"].predict(arr)
            decoded = decode_predictions(preds, top=3)[0]
            st.write("Model top predictions (ImageNet labels):")
            for cid, name, prob in decoded:
                st.write(f"- {name}: {prob:.3f}")

            st.info("Note: Replace this block with a medically trained mammogram classifier for real predictions.")
        except Exception as e:
            st.error(f"Mammogram demo error: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Project: OncoScan AI — OCR + Tabular ML + SHAP")
st.sidebar.write("Tip: For best OCR results, use a clear screenshot with labels like 'mean radius: 12.3' or a neat table.")
