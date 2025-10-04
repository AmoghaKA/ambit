import { useState } from 'react'
import './styles.css'

// Placeholder imports (to be implemented)
import FeatureForm from './components/FeatureForm.jsx'
import CsvUpload from './components/CsvUpload.jsx'
import ResultCard from './components/ResultCard.jsx'

function App() {
  // State for single prediction
  const [singleLoading, setSingleLoading] = useState(false)
  const [singleError, setSingleError] = useState(null)
  const [singleResult, setSingleResult] = useState(null)

  // State for batch prediction
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const [batchResult, setBatchResult] = useState(null)

  return (
    <div className="app-container">
      <header>
        <h1>Breast Cancer Predictor</h1>
      </header>

      <section className="card">
        <h2>Single Prediction</h2>
        <FeatureForm
          loading={singleLoading}
          error={singleError}
          result={singleResult}
          setLoading={setSingleLoading}
          setError={setSingleError}
          setResult={setSingleResult}
        />
        {singleResult && (
          <ResultCard result={singleResult} />
        )}
      </section>

      <section className="card">
        <h2>Batch CSV</h2>
        <CsvUpload
          loading={batchLoading}
          error={batchError}
          result={batchResult}
          setLoading={setBatchLoading}
          setError={setBatchError}
          setResult={setBatchResult}
        />
        {batchResult && (
          <ResultCard result={batchResult} />
        )}
      </section>

      <footer>
        Label 1 = malignant (per current model)
      </footer>
    </div>
  )
}

export default App
