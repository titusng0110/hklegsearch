import { useState, useRef } from 'react';
import './App.css';

function App() {
  const [payload, setPayload] = useState('');
  const [status, setStatus] = useState<{ message: string; type?: 'error' | 'loading' | 'success' }>({ message: '' });
  const [results, setResults] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const throttleTimerRef = useRef<number | null>(null);

  const handleSubmit = async () => {
    if (throttleTimerRef.current) return;
    
    // Clear previous results
    setResults([]);
    
    if (!payload.trim()) {
      setStatus({ message: 'Please enter a query.', type: 'error' });
      return;
    }

    // Set throttle
    setIsSubmitting(true);
    throttleTimerRef.current = window.setTimeout(() => {
      throttleTimerRef.current = null;
      setIsSubmitting(false);
    }, 500);

    setStatus({ message: 'Processing your request...', type: 'loading' });

    try {
      const response = await fetch(`/api/?payload=${encodeURIComponent(payload.trim())}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText);
      }
      
      const data = await response.json();
      
      setStatus({ message: 'Request successful!', type: 'success' });
      
      if (data?.texts) {
        setResults(data.texts);
      }
    } catch (err) {
      setStatus({ message: `Error: ${err instanceof Error ? err.message : 'Unknown error'}`, type: 'error' });
    }
  };

  return (
    <div className="container">
      <h1>Query Search</h1>
      <textarea 
        value={payload}
        onChange={(e) => setPayload(e.target.value)}
        maxLength={2048} 
        placeholder="Type your query here..." 
      />
      <button 
        onClick={handleSubmit} 
        disabled={isSubmitting}
      >
        Submit
      </button>
      
      <div className="status">
        {status.message && (
          <span className={status.type}>{status.message}</span>
        )}
      </div>
      
      {results.length > 0 && (
        <div className="results">
          <h2>Results:</h2>
          <ul>
            {results.map((text, idx) => (
              <li key={idx}>
                <strong>Document {idx + 1}:</strong> {text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;