import React, { useState, useRef, useEffect } from 'react';
import styles from './styles.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface Citation {
  module: string;
  chapter: string;
  section: string;
  content_preview: string;
  score: number;
}

interface ChatResponse {
  answer: string;
  citations: Citation[];
  context_used: number;
}

export default function Chatbot(): JSX.Element {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Handle text selection
  useEffect(() => {
    const handleTextSelection = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim();
      if (text && text.length > 3) {
        setSelectedText(text);
      }
    };

    document.addEventListener('mouseup', handleTextSelection);
    return () => document.removeEventListener('mouseup', handleTextSelection);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Use environment variable for backend URL, fallback to localhost for development
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          conversation_history: messages,
          max_results: 5,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data: ChatResponse = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer,
      };

      setMessages(prev => [...prev, assistantMessage]);
      setCitations(data.citations);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please make sure the backend server is running.',
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const askAboutSelection = () => {
    setInput(`Explain: "${selectedText}"`);
    setIsOpen(true);
    setSelectedText('');
  };

  return (
    <>
      {/* Text Selection Popup */}
      {selectedText && !isOpen && (
        <div className={styles.selectionPopup}>
          <button
            className={styles.askButton}
            onClick={askAboutSelection}
            title="Ask AI about this"
          >
            🤖 Ask about this
          </button>
        </div>
      )}

      {/* Floating Chat Button */}
      <button
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chat"
      >
        {isOpen ? '✕' : '💬'}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h3>AI Tutor</h3>
            <p>Ask me anything about ROS 2, Humanoid Robotics, or Physical AI!</p>
          </div>

          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.emptyState}>
                <p>👋 Hi! I'm your AI tutor.</p>
                <p>Try asking:</p>
                <ul>
                  <li>"How do I create a ROS 2 node?"</li>
                  <li>"What is the ZMP in bipedal walking?"</li>
                  <li>"Explain vision-language models"</li>
                </ul>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`${styles.message} ${
                  msg.role === 'user' ? styles.userMessage : styles.assistantMessage
                }`}
              >
                <div className={styles.messageContent}>{msg.content}</div>
              </div>
            ))}

            {loading && (
              <div className={`${styles.message} ${styles.assistantMessage}`}>
                <div className={styles.loadingDots}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Citations */}
          {citations.length > 0 && (
            <div className={styles.citations}>
              <details>
                <summary>📚 Sources ({citations.length})</summary>
                <ul>
                  {citations.map((citation, idx) => (
                    <li key={idx}>
                      <strong>{citation.module}</strong> - {citation.chapter} - {citation.section}
                      <br />
                      <small>{citation.content_preview}</small>
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          )}

          {/* Input Area */}
          <div className={styles.inputContainer}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question..."
              rows={2}
              disabled={loading}
              className={styles.input}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !input.trim()}
              className={styles.sendButton}
            >
              {loading ? '...' : '→'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
