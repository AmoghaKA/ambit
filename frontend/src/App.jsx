import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useNavigate } from 'react-router-dom'
import './styles.css'
import CancerDetection from './pages/CancerDetection'

function LandingPage() {
  const navigate = useNavigate()
  
  return (
    <div className="custom-landing-bg">
      <header className="custom-landing-header">
        <div className="custom-logo-row">
          <div className="custom-logo-circle">
            <img src="/rib.jpg" alt="OncoScan logo" className="custom-logo-img" />
          </div>
          <span className="custom-logo-text">OncoScan AI</span>
        </div>
        <nav className="custom-nav">
          <Link to="/breast" className="custom-nav-link">About Breast Cancer</Link>
          <Link to="/about" className="custom-nav-link">About Us</Link>
          <Link to="/contact" className="custom-nav-link">Contact Us</Link>
        </nav>
      </header>
      <main className="custom-landing-main">
        <section className="custom-landing-content">
          <h1 className="custom-landing-title">
            Early Detection,<br />
            Early Prevention<br />
          </h1>
          <button className="custom-landing-cta" onClick={() => navigate('/detect')}>
            Try Now <span className="custom-cta-arrow">→</span>
          </button>
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
              <li><Link to="/about">About Us</Link></li>
              <li><Link to="/detect">How It Works</Link></li>
              <li><a href="#">Pricing</a></li>
              <li><Link to="/contact">Contact</Link></li>
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

function AboutPage() {
  return (
    <div className="custom-landing-bg">
      <header className="custom-landing-header">
        <div className="custom-logo-row">
          <div className="custom-logo-circle">
            <img src="/rib.jpg" alt="OncoScan logo" className="custom-logo-img" />
          </div>
          <span className="custom-logo-text">OncoScan AI</span>
        </div>
        <nav className="custom-nav">
          <Link to="/" className="custom-nav-link">Home</Link>
          <Link to="/detect" className="custom-nav-link">Detect</Link>
          <Link to="/contact" className="custom-nav-link">Contact</Link>
        </nav>
      </header>
      <main className="custom-landing-main">
        <div className="card">
          <h1>About OncoScan AI</h1>
          <p>OncoScan AI is a cutting-edge medical technology platform that uses artificial intelligence to assist in the early detection of breast cancer. Our advanced machine learning algorithms analyze medical images and data to provide accurate, fast, and reliable predictions.</p>
          <h2>Our Mission</h2>
          <p>To make cancer detection more accessible, accurate, and early through the power of artificial intelligence, ultimately saving lives and improving healthcare outcomes.</p>
          <h2>Technology</h2>
          <p>We use state-of-the-art deep learning models, computer vision, and natural language processing to analyze medical reports, images, and patient data with unprecedented accuracy.</p>
        </div>
      </main>
    </div>
  )
}

function ContactPage() {
  return (
    <div className="custom-landing-bg">
      <header className="custom-landing-header">
        <div className="custom-logo-row">
          <div className="custom-logo-circle">
            <img src="/rib.jpg" alt="OncoScan logo" className="custom-logo-img" />
          </div>
          <span className="custom-logo-text">OncoScan AI</span>
        </div>
        <nav className="custom-nav">
          <Link to="/" className="custom-nav-link">Home</Link>
          <Link to="/about" className="custom-nav-link">About</Link>
          <Link to="/detect" className="custom-nav-link">Detect</Link>
        </nav>
      </header>
      <main className="custom-landing-main">
        <div className="card">
          <h1>Contact Us</h1>
          <p>Get in touch with our team for any questions, support, or partnership opportunities.</p>
          <div className="contact-info">
            <h3>Email</h3>
            <p>contact@oncoscan.ai</p>
            <h3>Phone</h3>
            <p>+1 (555) 123-4567</p>
            <h3>Address</h3>
            <p>123 Medical Tech Drive<br />Healthcare City, HC 12345</p>
          </div>
        </div>
      </main>
    </div>
  )
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/detect" element={<CancerDetection />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/contact" element={<ContactPage />} />
        <Route path="/breast" element={<AboutPage />} />
      </Routes>
    </Router>
  )
}

export default App
