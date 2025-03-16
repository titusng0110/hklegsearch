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
  const [currentKeywords, setCurrentKeywords] = useState<string[]>([]); // New state for keywords
  const throttleTimerRef = useRef<number | null>(null);

  // Function to process text and get stemmed keywords
  const processQuery = (text: string): string[] => {
    // Split text into words, convert to lowercase, and remove punctuation
    const words = text.toLowerCase().split(/\W+/);
    
    // Filter out stop words and empty strings, then stem remaining words
    return words
      .filter(word => word && !stopWords.includes(word))
      .map(word => stemmer.stemWord(word));
  };

  // Function to highlight keywords in text
  const highlightText = (text: string, keywords: string[]) => {
    // Create a copy of the text to work with
    let highlightedText = text;
    
    // Split the text into words while preserving spaces and punctuation
    const tokens = highlightedText.split(/(\W+)/);
    
    // Process each token
    const processedTokens = tokens.map(token => {
      // Skip spaces and punctuation
      if (token.trim() === '') return token;
      
      // Stem the current word
      const stemmedToken = stemmer.stemWord(token.toLowerCase());
      
      // If the stemmed word is in our keywords, highlight the original word
      if (keywords.includes(stemmedToken)) {
        return `<mark>${token}</mark>`;
      }
      return token;
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

    // Process keywords when submitting
    const newKeywords = processQuery(payload);
    setCurrentKeywords(newKeywords);

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
                <strong>Document {idx + 1}:</strong>{' '}
                {highlightText(text, currentKeywords)} {/* Using currentKeywords instead */}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;