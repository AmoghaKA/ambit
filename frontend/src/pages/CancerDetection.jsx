import React, { useState } from "react";
import "../styles/CancerDetection.css";
import UploadSection from "../components/UploadSection";
import axios from "axios";

function CancerDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [ocrText, setOcrText] = useState("");
  const [features, setFeatures] = useState(null);
  const [shapPlotUrl, setShapPlotUrl] = useState(null);
  const [valuesText, setValuesText] = useState("");

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please upload a report image first!");
      return;
    }

  const formData = new FormData();
  // backend expects field name 'file' (UploadFile param name in FastAPI)
  formData.append("file", selectedFile);

    setLoading(true);
    setPrediction(null);
    setOcrText("");
    setFeatures(null);
    setShapPlotUrl(null);

    try {
      // Let the browser set the Content-Type (with boundary)
      const res = await axios.post(`${API_BASE}/predict_image`, formData);

      const data = res.data;
      setPrediction({ result: data.label || "Unknown", confidence: data.probability ? (data.probability * 100).toFixed(2) : null });
      setOcrText(data.ocr_text || "");
      setFeatures(data.features || null);
      setShapPlotUrl(data.shap_plot_url || null);
    } catch (err) {
      console.error(err);
      // If server returned JSON with detail or message, show it
      const serverMsg = err?.response?.data?.detail || err?.response?.data?.message || err?.message;
      alert("Prediction failed: " + (serverMsg || "Check backend server and console."));
    } finally {
      setLoading(false);
    }
  };

  const handleValuesPredict = async () => {
    const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const raw = valuesText || "";
    const nums = raw.split(/[\s,]+/).filter(Boolean).map(Number);
    if (nums.some(isNaN)) {
      alert("Please enter only numbers separated by spaces or commas.");
      return;
    }
    if (!(nums.length === 10 || nums.length === 30)) {
      alert("Please provide exactly 10 or 30 numeric values.");
      return;
    }
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/predict_values`, { values: nums });
      const data = res.data;
      setPrediction({ result: data.label || "Unknown", confidence: data.probability ? (data.probability * 100).toFixed(2) : null });
      setFeatures(data.features || null);
    } catch (err) {
      console.error(err);
      const serverMsg = err?.response?.data?.detail || err?.response?.data?.message || err?.message;
      alert("Prediction failed: " + (serverMsg || "Check backend server."));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="cancer-container">
      <h1 className="cancer-title">🧠 Breast Cancer Detection</h1>
      <p className="cancer-subtitle">
        Upload your biopsy or mammogram report to analyze using AI.
      </p>

      <UploadSection onFileChange={handleFileChange} selectedFile={selectedFile} />

      <button className="predict-btn" onClick={handlePredict} disabled={loading}>
        {loading ? "Analyzing..." : "Predict"}
      </button>

      <div className="manual-values">
        <h3>Or paste 10 or 30 numeric values (space/comma separated)</h3>
        <textarea
          placeholder="Enter 10 or 30 numbers..."
          value={valuesText}
          onChange={(e) => setValuesText(e.target.value)}
          rows={4}
          style={{ width: "100%" }}
        />
        <button className="predict-btn" onClick={handleValuesPredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict from values"}
        </button>
      </div>

      {prediction && (
        <div className={`result-card ${prediction.result.toLowerCase()}`}>
          <h2>{prediction.result}</h2>
          <p>Confidence: {prediction.confidence}%</p>
        </div>
      )}

      {ocrText && (
        <div className="ocr-text-card">
          <h3>OCR Extracted Text (first 500 chars)</h3>
          <pre>{ocrText.slice(0, 500)}</pre>
        </div>
      )}

      {features && (
        <div className="features-card">
          <h3>Extracted Features</h3>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(features).map(([key, value]) => (
                <tr key={key}>
                  <td>{key}</td>
                  <td>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {shapPlotUrl && (
        <div className="shap-card">
          <h3>SHAP Explanation</h3>
          <img src={shapPlotUrl} alt="SHAP Plot" className="shap-plot" />
        </div>
      )}
    </div>
  );
}

export default CancerDetection;
