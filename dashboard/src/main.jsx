import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import './index.css'
import App from './App.jsx'
import ComponentTest from './ComponentTest.jsx'

// Simple navigation for testing
function TestNav() {
  return (
    <div style={{ padding: '20px', background: '#1a1a2e', borderBottom: '1px solid #333' }}>
      <Link to="/" style={{ color: '#6366f1', marginRight: '20px', textDecoration: 'none' }}>
        Original App
      </Link>
      <Link to="/test-components" style={{ color: '#6366f1', textDecoration: 'none' }}>
        Test Components
      </Link>
    </div>
  );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <TestNav />
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/test-components" element={<ComponentTest />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
