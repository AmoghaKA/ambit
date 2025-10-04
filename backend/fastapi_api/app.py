import os, io, time, traceback
import numpy as np
import pandas as pd
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
try:
    import cloudpickle
except ImportError:
    cloudpickle = None
# optional OCR
try:
    from PIL import Image
    import pytesseract
except Exception:
    Image = None
    pytesseract = None

MODEL_PATH = os.getenv("MODEL_PATH", "model/breast_cancer_model.pkl")

FEATURE_NAMES = [
    "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
    "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
    "radius error","texture error","perimeter error","area error","smoothness error",
    "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
    "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
    "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
]

app = FastAPI(title="Tumor Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class PredictRequest(BaseModel):
    features: List[float] = Field(min_length=30, max_length=30)

_model = None
_model_error = None

def load_model():
    global _model, _model_error
    try:
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
            raise FileNotFoundError(f"MODEL_PATH '{MODEL_PATH}' missing or empty.")
        _model = joblib.load(MODEL_PATH)
        _model_error = None
        print(f"✅ Model loaded via joblib from {MODEL_PATH}")
    except Exception as e1:
        if cloudpickle is None:
            _model = None
            _model_error = f"joblib failed: {e1}\nInstall cloudpickle to try fallback."
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = cloudpickle.load(f)
            _model_error = None
            print(f"✅ Model loaded via cloudpickle from {MODEL_PATH}")
        except Exception as e2:
            _model = None
            _model_error = (
                "joblib failed:\n" + "".join(traceback.format_exception(type(e1), e1, e1.__traceback__)) +
                "\ncloudpickle failed:\n" + "".join(traceback.format_exception(type(e2), e2, e2.__traceback__))
            )

def ensure_model_loaded():
    if _model is None and _model_error is None:
        load_model()
    if _model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {_model_error}")

@app.get("/health")
def health():
    size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None and _model_error is None,
        "model_error": _model_error,
        "model_size_bytes": size
    }

@app.post("/reload_model")
def reload_model():
    load_model()
    if _model is None:
        raise HTTPException(status_code=500, detail=f"Reload failed: {_model_error}")
    return {"reloaded": True, "model_path": MODEL_PATH}

@app.post("/predict")
def predict(req: PredictRequest):
    ensure_model_loaded()
    t0 = time.time()
    X = np.array(req.features, dtype=float).reshape(1, -1)
    if not hasattr(_model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model has no predict_proba().")
    proba = _model.predict_proba(X)[0]
    prob_benign = float(proba[1]) if len(proba) == 2 else float(proba)
    prob_malignant = 1.0 - prob_benign
    label = int(prob_malignant >= 0.5)
    return {"prob_malignant": prob_malignant, "label": label, "latency_ms": int((time.time()-t0)*1000)}

@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    ensure_model_loaded()
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Please upload a CSV/TXT file.")
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    try:
        X = df[FEATURE_NAMES]
    except KeyError:
        raise HTTPException(status_code=400, detail=f"CSV must contain required columns: {FEATURE_NAMES}")
    if not hasattr(_model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Loaded model has no predict_proba().")
    proba_b = _model.predict_proba(X)[:, 1]
    prob_m = 1.0 - proba_b
    label = (prob_m >= 0.5).astype(int)
    preview = df.head(5).copy()
    preview["prob_malignant"] = prob_m[:5]
    preview["label"] = label[:5]
    return {"count": int(len(df)), "positives": int(label.sum()), "results_preview": preview.to_dict(orient="records")}


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    """Accepts an image (jpg/png) containing either labeled values or a table. Runs OCR and attempts to extract the 10 mean_* features used by the Streamlit demo's 10-feature pipeline.
    Response: { label, probability, ocr_text, features }
    """
    ensure_model_loaded()
    if Image is None or pytesseract is None:
        raise HTTPException(status_code=500, detail="OCR dependencies not installed (Pillow/pytesseract)")
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        raise HTTPException(status_code=400, detail="Please upload an image file for OCR")
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        ocr_text = pytesseract.image_to_string(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # try to parse numbers from OCR text using simple regex similar to Streamlit helper
    import re
    NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
    nums = NUM_RE.findall(ocr_text)

    # If model expects 30 features (original pipeline), raise informative message; else if the loaded model has steps and expects 10 features, handle that
    # We'll try to support both: if 30 numbers found => use /predict endpoint; if 10 numbers => use the model directly for the 10-feature pipeline
    if len(nums) >= 30:
        # map first 30 numbers to features and call existing predict route
        arr = [float(x) for x in nums[:30]]
        # reuse existing predict which returns prob_malignant and label int
        resp = await predict(PredictRequest(features=arr))
        # convert to consistent response: label string and probability
        lbl = "Malignant" if resp.get("label", 0) == 1 else "Benign"
        prob = resp.get("prob_malignant", None)
        return {"label": lbl, "probability": prob, "ocr_text": ocr_text, "features": None}

    # handle 10-feature pipeline
    if len(nums) >= 10:
        vals10 = [float(x) for x in nums[:10]]
        # build input vector expected by pipeline: many pipelines expect DataFrame; try both
        try:
            X = np.array(vals10, dtype=float).reshape(1, -1)
            if hasattr(_model, "predict_proba"):
                proba = _model.predict_proba(X)[0]
                # assume class ordering where index 1 corresponds to malignant probability if present
                if len(proba) == 2:
                    prob_malignant = float(proba[1])
                else:
                    prob_malignant = float(proba)
            else:
                raise HTTPException(status_code=500, detail="Loaded model has no predict_proba()")
            lbl = "Malignant" if prob_malignant >= 0.5 else "Benign"
            features_map = {f"feat_{i}": float(vals10[i]) for i in range(len(vals10))}
            return {"label": lbl, "probability": prob_malignant, "ocr_text": ocr_text, "features": features_map}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # otherwise return OCR text and notice
    return {"label": None, "probability": None, "ocr_text": ocr_text, "features": None, "note": "Not enough numeric values found (need 10 or 30)."}
