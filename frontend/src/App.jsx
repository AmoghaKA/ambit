import { useState } from 'react'
import './styles.css'

function LandingPage({ onTryNow }) {
  return (
    <div className="custom-landing-bg">
      <header className="custom-landing-header">
        <div className="custom-logo-row">
          <div className="custom-logo-circle">
            {/* Placeholder for logo icon */}
            <img src="/rib.jpg" alt="OncoScan logo" className="custom-logo-img" />
          </div>
          <span className="custom-logo-text">OncoScan AI</span>
        </div>
        <nav className="custom-nav">
        <a href="breast" className="custom-nav-link">About Breast Cancer</a>
          <a href="about" className="custom-nav-link">About Us</a>
          <a href="us" className="custom-nav-link">Contact Us</a>
          
        </nav>
      </header>
      <main className="custom-landing-main">
        <section className="custom-landing-content">
          <h1 className="custom-landing-title">
            Early Detection,<br />
            Early Prevention<br />
           
          </h1>
          <button className="custom-landing-cta" onClick={onTryNow}>Try Now <span className="custom-cta-arrow">→</span></button>
          <div className="custom-landing-features">
            <div className="custom-feature">
              <span className="custom-feature-icon">🛡️</span>
              <span className="custom-feature-label">Secure & Private</span>
            </div>
            <div className="custom-feature">
              <span className="custom-feature-icon">🤖</span>
              <span className="custom-feature-label">AI Powered Accuracy</span>
            </div>
            <div className="custom-feature">
              <span className="custom-feature-icon">💡</span>
              <span className="custom-feature-label">User Friendly Interface</span>
            </div>
          </div>
        </section>
        <section className="custom-landing-visuals">
          <div className="custom-women-container">
            <img 
              src="/women-image.jpg" 
              alt="Two professional women" 
              className="custom-women-image"
            />
          </div>
        </section>
      </main>
      <footer className="custom-landing-footer">
        <div className="footer-content">
          <div className="footer-section">
            <h3>OncoScan AI</h3>
            <p>Empowering early detection through advanced AI technology for breast cancer diagnosis.</p>
            <div className="social-links">
              <a href="#" className="social-link">LinkedIn</a>
              <a href="#" className="social-link">Twitter</a>
              <a href="#" className="social-link">Facebook</a>
            </div>
          </div>
          <div className="footer-section">
            <h4>Quick Links</h4>
            <ul>
              <li><a href="#">About Us</a></li>
              <li><a href="#">How It Works</a></li>
              <li><a href="#">Pricing</a></li>
              <li><a href="#">Contact</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Resources</h4>
            <ul>
              <li><a href="#">Help Center</a></li>
              <li><a href="#">Documentation</a></li>
              <li><a href="#">API Reference</a></li>
              <li><a href="#">Support</a></li>
            </ul>
          </div>
          <div className="footer-section">
            <h4>Legal</h4>
            <ul>
              <li><a href="#">Privacy Policy</a></li>
              <li><a href="#">Terms of Service</a></li>
              <li><a href="#">Cookie Policy</a></li>
              <li><a href="#">Disclaimer</a></li>
            </ul>
          </div>
        </div>
        <div className="footer-bottom">
          <p>&copy; 2025 OncoScan AI. All rights reserved.</p>
          <p>Made with ❤️ for better healthcare</p>
        </div>
      </footer>
    </div>
  )
}

function PredictionPage({ onBack }) {
  return (
    <div className="prediction-page">
      <header className="prediction-header">
        <button className="back-button" onClick={onBack}>← Back</button>
        <h1>Breast Cancer Diagnosis</h1>
      </header>
      <main className="prediction-main">
        <div className="prediction-card">
          <h2>Upload Your Data</h2>
          <p>Please upload your medical data for analysis</p>
          <div className="upload-area">
            <input type="file" accept=".csv,.xlsx,.jpg,.jpeg,.png" id="file-input" style={{display: 'none'}} />
            <button className="upload-btn" onClick={() => document.getElementById('file-input').click()}>Choose File</button>
          </div>
        </div>
        <div className="prediction-card">
          <h2>Prediction Results</h2>
          <p>Your results will appear here after analysis</p>
        </div>
      </main>
    </div>
  )
}

function App() {
  const [currentPage, setCurrentPage] = useState('landing')

  if (currentPage === 'landing') {
    return <LandingPage onTryNow={() => setCurrentPage('prediction')} />
  }

  if (currentPage === 'prediction') {
    return <PredictionPage onBack={() => setCurrentPage('landing')} />
  }

  return <LandingPage onTryNow={() => setCurrentPage('prediction')} />
}

export default App
