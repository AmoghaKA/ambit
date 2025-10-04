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

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please upload a report image first!");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setLoading(true);
    setPrediction(null);
    setOcrText("");
    setFeatures(null);
    setShapPlotUrl(null);

    try {
      const res = await axios.post("http://localhost:8000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const data = res.data;
      setPrediction({ result: data.label, confidence: (data.probability * 100).toFixed(2) });
      setOcrText(data.ocr_text || "");
      setFeatures(data.features || null);
      setShapPlotUrl(data.shap_plot_url || null);
    } catch (err) {
      console.error(err);
      alert("Prediction failed! Check backend server.");
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
