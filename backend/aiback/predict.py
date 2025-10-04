import os, io, re, base64
from io import BytesIO
from flask import Blueprint, request, jsonify
import joblib, numpy as np, pandas as pd
from PIL import Image
import cv2, pytesseract, easyocr
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

bp = Blueprint("predict", _name_)

# --------- CONFIG ----------
MODEL_PATH = os.path.join(os.path.dirname(_file_), "model_pipeline_10.pkl")
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # update if needed
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# load model once
model = joblib.load(MODEL_PATH)

# --------- HELPERS ----------
def ocr_from_bytes(img_bytes):
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)
    arr = cv2.fastNlMeansDenoising(arr, h=10)
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(th)
    except Exception:
        text = ""
    if len(text.strip()) < 8:
        # fallback easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        res = reader.readtext(np.array(pil))
        text = "\n".join([r[1] for r in res])
    return pil, text

def extract_numbers(text, n=10):
    found = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    vals = [float(x) for x in found[:n]]
    # pad with 0s if not enough numbers
    while len(vals) < n:
        vals.append(0.0)
    return vals

def get_expected_feature_names():
    # try model.feature_names_in_ (scikit-learn >=0.24)
    try:
        return list(model.feature_names_in_)
    except Exception:
        pass
    # if it's a pipeline, try to get names from preprocessor if available
    if isinstance(model, Pipeline):
        try:
            pre = model[:-1]
            if hasattr(pre, "get_feature_names_out"):
                return list(pre.get_feature_names_out())
        except Exception:
            pass
    # fallback to simple list of 10 generic names
    return [f"f{i}" for i in range(10)]
# --------- PREDICT ROUTE ----------
@bp.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "send form-data with file field named 'file'"}), 400

    f = request.files["file"]
    img_bytes = f.read()

    try:
        pil_img, ocr_text = ocr_from_bytes(img_bytes)
    except Exception as e:
        return jsonify({"error": "OCR failed", "detail": str(e)}), 500

    values = extract_numbers(ocr_text, n=10)  # adjust n if your model needs >10
    feature_names = get_expected_feature_names()
    # use first 10 feature names (or fallback)
    cols = feature_names[:10] if len(feature_names) >= 10 else [f"f{i}" for i in range(10)]
    df = pd.DataFrame([values], columns=cols)

    # If model expects specific columns (feature_names_in_), align them
    try:
        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            for c in needed:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[needed]
    except Exception:
        pass

 # Predict
    try:
        proba = float(model.predict_proba(df)[0][1])
        label = "Malignant" if proba >= 0.5 else "Benign"
    except Exception as e:
        return jsonify({"error": "prediction failed", "detail": str(e)}), 500

    # SHAP explainability (best-effort)
    shap_img_b64 = None
    shap_values = None
    try:
        # if pipeline: separate preprocessor and classifier
        if isinstance(model, Pipeline):
            pre = model[:-1]
            clf = model[-1]
            X_trans = pre.transform(df)
        else:
            clf = model
            X_trans = df

        # prefer TreeExplainer if possible (fast) otherwise fallback
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer(X_trans)
        except Exception:
            explainer = shap.Explainer(clf, X_trans)
            sv = explainer(X_trans)

        # try to extract values & create a quick matplotlib bar plot
        vals = getattr(sv, "values", np.array(sv))
        # If vector shape: collapse to absolute sum per feature
        if vals.ndim == 3:
            # (samples, outputs, features) -> take class 1 if present else sum
            vals2 = np.abs(vals[0]).sum(axis=0)
        elif vals.ndim == 2:
            vals2 = np.abs(vals[0])
        else:
            vals2 = np.abs(vals)
        shap_values = vals.tolist()
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(len(vals2)), vals2)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
        ax.set_title("SHAP importance (abs)")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        shap_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # SHAP failed — do not crash prediction
        shap_img_b64 = None

    return jsonify({
        "label": label,
        "confidence": proba,
        "extracted_values": values,
        "ocr_text": ocr_text,
        "shap_image": shap_img_b64,
        "shap_values": shap_values
    })