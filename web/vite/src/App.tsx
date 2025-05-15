import { useState, useRef } from 'react';
import './App.css';
import SnowballEn from './SnowballEn.js';

function App() {
  const stemmer = new SnowballEn();
  const stopWords = ["ever", "hardly", "hence", "into", "nor", "were", "viz", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "could", "did", "do", "does", "e.g.", "from", "had", "has", "have", "having", "he", "her", "here", "hereby", "herein", "hereof", "hereon", "hereto", "herewith", "him", "his", "however", "i.e.", "if", "is", "it", "its", "me", "of", "on", "onto", "or", "our", "really", "said", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "thereby", "therefore", "therefrom", "therein", "thereof", "thereon", "thereto", "therewith", "these", "they", "this", "those", "thus", "to", "too", "unto", "us", "very", "was", "we", "what", "when", "where", "whereby", "wherein", "whether", "which", "who", "whom", "whose", "why", "with", "would", "you"]; // lexisnexis stop words
  
  const [payload, setPayload] = useState('');
  const [status, setStatus] = useState<{ message: string; type?: 'error' | 'loading' | 'success' }>({ message: '' });
  const [results, setResults] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [currentKeywords, setCurrentKeywords] = useState<string[]>([]);
  const throttleTimerRef = useRef<number | null>(null);

  const processQuery = (text: string): string[] => {
    const words = text.toLowerCase().split(/\W+/);
    return words
      .filter(word => word && !stopWords.includes(word))
      .map(word => stemmer.stemWord(word));
  };

  const highlightText = (text: string, keywords: string[]) => {
    let highlightedText = text;
    const tokens = highlightedText.split(/(\W+)/);
    
    const processedTokens = tokens.map(token => {
      // Handle whitespace-only tokens (including newlines)
      if (token.trim() === '') {
        return token.replace(/\n/g, '<br />');
      }
      
      const stemmedToken = stemmer.stemWord(token.toLowerCase());
      let processedToken = token;
      
      if (keywords.includes(stemmedToken)) {
        processedToken = `<mark>${token}</mark>`;
      }
      
      // Replace newlines with <br /> in all tokens
      processedToken = processedToken.replace(/\n/g, '<br />');
      return processedToken;
    });

    return (
      <span dangerouslySetInnerHTML={{ __html: processedTokens.join('') }} />
    );
  };

  const handleSubmit = async () => {
    if (throttleTimerRef.current) return;
    
    setResults([]);
    
    if (!payload.trim()) {
      setStatus({ message: 'Please enter a query.', type: 'error' });
      return;
    }

    const newKeywords = processQuery(payload);
    setCurrentKeywords(newKeywords);

    setIsSubmitting(true);
    throttleTimerRef.current = window.setTimeout(() => {
      throttleTimerRef.current = null;
      setIsSubmitting(false);
    }, 500);

    setStatus({ message: 'Processing your request...', type: 'loading' });

    try {
      const response = await fetch(`/api/?type=${encodeURIComponent('leg')}&payload=${encodeURIComponent(payload.trim())}`);
      
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
        placeholder="E.g. What are the elements of drink driving?"
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
                <strong>Document {idx + 1}:</strong><br />
                {highlightText(text, currentKeywords)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;