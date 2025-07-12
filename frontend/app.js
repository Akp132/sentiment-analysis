// app.js
const { useState, useEffect, useCallback } = React;

const ConfidenceMeter = ({ score, label }) => {
  // Convert score to percentage
  const percentage = score * 100;
  
  // Get confidence level
  const getConfidenceLevel = (score) => {
    if (score < 0.6) return "Low";
    if (score < 0.8) return "Medium";
    return "High";
  };

  return (
    <div className="confidence-meter">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
        <span className={`sentiment-label ${label.toLowerCase()}`}>
          {label}
        </span>
        <span className="confidence-value">
          {percentage.toFixed(1)}% Confidence
        </span>
      </div>
      <div className="meter-container">
        <div 
          className="meter-fill"
          style={{ width: `${percentage}%` }}
        />
        {/* Confidence markers */}
        <div className="meter-marker" style={{ left: '60%' }} />
        <div className="meter-marker" style={{ left: '80%' }} />
      </div>
      <div className="meter-labels">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
      </div>
    </div>
  );
};

function App() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [liveResult, setLiveResult] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const backendUrl = window.BACKEND_URL || "http://localhost:8000";

  // Dark mode effect
  useEffect(() => {
    // Check system preference
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setIsDarkMode(prefersDark);
    
    document.documentElement.setAttribute("data-theme", prefersDark ? "dark" : "light");
  }, []);

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.setAttribute("data-theme", !isDarkMode ? "dark" : "light");
  };

  // Debounce function
  const debounce = (func, wait) => {
    let timeout;
    return (...args) => {
      clearTimeout(timeout);
      timeout = setTimeout(() => func(...args), wait);
    };
  };

  // Live inference
  const performLiveInference = useCallback(
    debounce(async (text) => {
      if (!text.trim()) {
        setLiveResult(null);
        setIsTyping(false);
        return;
      }

      try {
        const response = await fetch(`${backendUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!response.ok) throw new Error("Request failed");
        const data = await response.json();
        setLiveResult(data);
        setIsTyping(false);
      } catch (err) {
        console.error("Live inference error:", err);
        setLiveResult(null);
      }
    }, 500),
    [backendUrl]
  );

  const handleTextChange = (e) => {
    const newText = e.target.value;
    setText(newText);
    if (newText.trim()) {
      setIsTyping(true);
      performLiveInference(newText);
    } else {
      setLiveResult(null);
      setIsTyping(false);
    }
  };

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error("Request failed");
      const data = await response.json();
      setResult(data);
      setLiveResult(null);
    } catch (err) {
      console.error(err);
      alert("Error during prediction");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <button className="theme-toggle" onClick={toggleTheme}>
        {isDarkMode ? "☀️ Light" : "🌙 Dark"}
      </button>
      <div className="container">
        <h2>Sentiment Analysis</h2>
        <textarea
          value={text}
          onChange={handleTextChange}
          placeholder="Enter text here..."
        />
        {isTyping && (
          <div className="live-indicator live-typing">
            Analyzing...
          </div>
        )}
        {liveResult && !result && !loading && (
          <div className="live-indicator">
            <ConfidenceMeter 
              score={liveResult.score} 
              label={liveResult.label}
            />
          </div>
        )}
        <button onClick={handlePredict} disabled={loading || !text.trim()}>
          {loading ? "Predicting..." : "Predict"}
        </button>
        {result && (
          <div className="result">
            <ConfidenceMeter 
              score={result.score} 
              label={result.label}
            />
          </div>
        )}
      </div>
    </>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />); 