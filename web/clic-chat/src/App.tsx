import React, { useState, useEffect, useRef } from 'react';
import './App.css';

type Message = { sender: 'user' | 'assistant'; text: string };
type Query = { query: string; option: string };

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'assistant', text: 'Hello, how can I help you?' }
  ]);
  const [input, setInput] = useState('');
  const [loadingPhase, setLoadingPhase] = useState<null | 'chat' | 'queries' | 'search' | 'advice'>(null);
  const [response1, setResponse1] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const inputRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loadingPhase && !response1) {
      inputRef.current?.focus();
    }
  }, [loadingPhase, response1]);

  // Auto‐scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send user message
  const handleSend = async () => {
    if (!input.trim() || loadingPhase) return;
    setError(null);

    const userText = input.trim();
    setMessages(m => [...m, { sender: 'user', text: userText }]);
    setInput('');
    setLoadingPhase('chat');

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userText })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      if (data.session_id) setSessionId(data.session_id);
      setMessages(m => [...m, { sender: 'assistant', text: data.reply }]);

      if (data.complete) {
        setResponse1(data.reply);
        await runQueries(data.session_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoadingPhase(null);
    }
  };

  // 1) /queries
  const runQueries = async (sid: string) => {
    setLoadingPhase('queries');
    try {
      const res = await fetch('/queries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid })
      });
      if (!res.ok) throw new Error(await res.text());
      const { queries, error: qe } = await res.json();
      if (qe) throw new Error(qe);

      // Emit queries as a chat message
      const formatted = queries
        .map((q: Query, i: number) => `${i + 1}. "${q.query}" (option: ${q.option})`)
        .join('\n');
      setMessages(m => [
        ...m,
        { sender: 'assistant', text: `🔍 Generated search queries:\n${formatted}` }
      ]);

      await runSearch(queries);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoadingPhase(null);
    }
  };

  // 2) /search
  const runSearch = async (qs: Query[]) => {
    setLoadingPhase('search');
    try {
      const res = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ queries: qs })
      });
      if (!res.ok) throw new Error(await res.text());
      const { search_results, error: se } = await res.json();
      if (se) throw new Error(se);

      // Emit search results as a chat message
      const formatted = search_results
        .map((results: string[], i: number) => {
          const header = `Results for query ${i + 1}:`;
          const items = results.map(r => `- ${r}`).join('\n');
          return `${header}\n${items}`;
        })
        .join('\n\n');
      setMessages(m => [
        ...m,
        { sender: 'assistant', text: `📄 Search results:\n${formatted}` }
      ]);

      await runAdvice(sessionId!, search_results);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      setLoadingPhase(null);
    }
  };

  // 3) /advice
  const runAdvice = async (sid: string, sr: string[][]) => {
    setLoadingPhase('advice');
    try {
      const res = await fetch('/advice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, search_results: sr })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      setMessages(m => [...m, { sender: 'assistant', text: data.advice }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoadingPhase(null);
    }
  };

  // Enter sends (no Shift+Enter)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="container">
      <h1>CLIC-Chat 3.0</h1>

      <span className="disclaimer">Disclaimer: This is a computer-generated response and does not constitute legal advice. For proper legal assistance, please consult a qualified legal professional.</span>

      <div className="chatWindow">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.sender}`}>
            {m.text.split('\n').map((line, idx) => (
              <p key={idx}>{line}</p>
            ))}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {error && <div className="status error">{error}</div>}
      {loadingPhase && (
        <div className="status loading">
          {{
            chat: 'Waiting for response…',
            queries: 'Generating queries…',
            search: 'Searching legal database…',
            advice: 'Drafting advice…'
          }[loadingPhase]}
        </div>
      )}

      <div className="inputRow">
        <textarea
          ref={inputRef}
          placeholder="Type your message..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={!!loadingPhase || !!response1}
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || !!loadingPhase || !!response1}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;