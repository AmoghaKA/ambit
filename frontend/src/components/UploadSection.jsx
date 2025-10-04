import React from "react";
import "../styles/CancerDetection.css";

function UploadSection({ onFileChange, selectedFile }) {
  return (
    <div className="upload-section">
      <label htmlFor="file-upload" className="upload-label">
        Choose Report
      </label>
      <input
        id="file-upload"
        type="file"
        accept="image/*,.pdf"
        onChange={onFileChange}
        className="upload-input"
      />

      {selectedFile && (
        <p className="file-name">
          📄 {selectedFile.name}
        </p>
      )}
    </div>
  );
}

export default UploadSection;
