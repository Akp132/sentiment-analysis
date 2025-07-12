# 📹 Demo Video

[Watch the demo here](https://drive.google.com/file/d/1x_wNb0BzwVfzDEm3do65d8JmeeEhLcXq/view?usp=sharing)

# Sentiment Analysis Microservice

This project provides an end-to-end, container-ready solution for **binary sentiment analysis**.
It contains:

1. **Python FastAPI backend** – loads a Hugging Face transformer and exposes a REST endpoint for inference.
2. **Fine-tuning CLI** – retrain the model on ## 8. Extending / Optional Enhancements



## 1. Quick Start (Docker)

```bash
git clone <repo> sentiment-analysis
cd sentiment-analysis

# Build and run the whole stack
# (first run downloads the model – can take ~1 minute)
docker-compose up --build
```

* Open **http://localhost:3000** – paste text and hit *Predict*.
* Backend is reachable on **http://localhost:8000** (OpenAPI docs at `/docs`).

Stop with **Ctrl-C**.

---

## 2. Project Structure

```
├── backend/             # FastAPI service & Dockerfile
│   ├── app.py           # Service entry-point
│   ├── requirements.txt # Python deps
│   └── model/           # Fine-tuned weights (populated after training)
├── frontend/            # Static React page served by Nginx
│   ├── index.html
│   └── app.js
├── finetune.py          # Stand-alone training script
├── docker-compose.yml   # Multi-service orchestration
└── README.md
```

---

## 3. API Documentation


### Endpoints

#### `POST /predict`
Analyzes the sentiment of provided text.

**Request:**
```http
POST /predict
Content-Type: application/json

{
  "text": "I absolutely loved it!"
}
```

**Request Schema:**
```json
{
  "text": {
    "type": "string",
    "description": "Text to analyze for sentiment",
    "required": true,
    "minLength": 1,
    "maxLength": 5000,
    "example": "I absolutely loved it!"
  }
}
```

**Successful Response (HTTP 200):**
```json
{
  "label": "POSITIVE",
  "score": 0.9876,
  "confidence_level": "High"
}
```

**Response Schema:**
```json
{
  "label": {
    "type": "string",
    "enum": ["POSITIVE", "NEGATIVE"],
    "description": "Predicted sentiment label"
  },
  "score": {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
    "description": "Confidence score for the prediction"
  },
  "confidence_level": {
    "type": "string",
    "enum": ["Low", "Medium", "High"],
    "description": "Human-readable confidence level"
  }
}
```

**Error Responses:**

```json
// HTTP 422 - Validation Error
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}

// HTTP 400 - Bad Request
{
  "detail": "Text is too long. Maximum length is 5000 characters."
}

// HTTP 500 - Internal Server Error
{
  "detail": "Model inference failed. Please try again."
}
```

#### `GET /health`
Health check endpoint for monitoring and load balancers.

**Response (HTTP 200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-07-13T10:30:00Z"
}
```

#### `GET /docs`
Interactive Swagger UI documentation (FastAPI auto-generated).

#### `GET /redoc`
Alternative ReDoc documentation interface.

#### `GET /openapi.json`
OpenAPI 3.0 specification in JSON format.

### Rate Limits
Currently no rate limits implemented. Recommended limits for production:
- **Development**: 100 requests/minute
- **Production**: 1000 requests/minute per API key

### Error Handling
All errors follow RFC 7807 Problem Details format:

| Status Code | Description | Example |
|-------------|-------------|---------|
| `200` | Success | Sentiment prediction returned |
| `400` | Bad Request | Invalid input format |
| `422` | Validation Error | Missing required fields |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Model loading failed |
| `503` | Service Unavailable | Server overloaded |

### SDKs and Examples

#### Python
```python
import requests

# Basic prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie is amazing!"}
)
result = response.json()
print(f"Sentiment: {result['label']} ({result['score']:.2f})")
```

#### JavaScript/Node.js
```javascript
// Using fetch API
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: 'This movie is amazing!'
  })
});

const result = await response.json();
console.log(`Sentiment: ${result.label} (${result.score.toFixed(2)})`);
```

#### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie is amazing!"}'
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Spec**: http://localhost:8000/openapi.json

---

## 4. Fine-tuning

Fine-tuning is **optional**. Provide a small JSONL dataset with one record per line:

```json
{"text": "Great product!", "label": "positive"}
{"text": "Worst experience.", "label": "negative"}
```

### Local Setup (Alternative to Docker)

#### Backend Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Run backend server
python app.py
```

#### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Open index.html in browser
open index.html  # macOS
# Or double-click the file on any OS
```

### Fine-tuning Commands

Run on CPU:

```bash
python finetune.py --data data.jsonl --epochs 3 --lr 2e-5 \
                   --lr_scheduler_type cosine --warmup_steps 500
```

Run on GPU (if available):

```bash
python finetune.py --data data.jsonl --epochs 3 --lr 2e-5 \
                   --lr_scheduler_type cosine --warmup_steps 500 --device cuda
```

Weights are saved to `backend/model/`. On the next API restart (`docker-compose up`) the service will automatically load the new weights.

### Performance Comparison: CPU vs GPU Fine-tuning

| Environment | Hardware | Dataset Size | Training Time | Batch Size | Memory Usage |
|-------------|----------|--------------|---------------|------------|--------------|
| **CPU** | Intel i7-1260P | 100 samples | ~70s | 8 | 4GB RAM |
| **CPU** | Intel i7-10700K | 1000 samples | ~8 minutes | 8 | 8GB RAM |
| **CPU** | Intel i7-10700K | 10K samples | ~45 minutes | 8 | 8GB RAM |
| **GPU** | NVIDIA RTX 3080 | 1000 samples | ~90s | 32 | 6GB VRAM |
| **GPU** | NVIDIA RTX 3080 | 10K samples | ~8 minutes | 32 | 6GB VRAM |
| **GPU** | NVIDIA RTX 3080 | 25K samples | ~15 minutes | 32 | 8GB VRAM |

#### GPU Advantages:
- **5-6x faster training** for larger datasets (>1K samples)
- **4x larger batch sizes** leading to better gradient estimates
- **Better memory efficiency** for transformer models
- **Parallel attention computation** optimized for GPU architecture

#### CPU Considerations:
- **Sufficient for small datasets** (<1K samples)
- **No specialized hardware required** - accessible to all developers
- **Lower power consumption** for small-scale fine-tuning
- **Acceptable inference speed** for real-time web applications

#### Recommendations:
- **Development/Prototyping**: CPU is adequate for initial experiments
- **Production fine-tuning**: GPU recommended for datasets >1K samples
- **Inference deployment**: CPU sufficient for most web applications
- **Large-scale training**: GPU essential for datasets >10K samples

---

## 5. Design Decisions

### Architecture Choices

#### Backend (FastAPI)
* **FastAPI** chosen for its speed, type hints & automatic OpenAPI documentation
* **Transformers pipeline** abstracts preprocessing + postprocessing for consistent results
* **DistilBERT SST-2** serves as a robust default English sentiment model with good performance/size ratio
* **Async endpoints** for non-blocking operations and better concurrency
* **CORS enabled** to allow frontend-backend communication from different origins
* **Error handling** with proper HTTP status codes and detailed error messages

#### Frontend (Vanilla React)
* **CDN-based React** for simplified deployment without complex build processes
* **Real-time inference** with debounced API calls for smooth user experience
* **Responsive design** using CSS Grid and Flexbox for modern layouts
* **Dark/Light mode toggle** with system preference detection
* **Interactive confidence meter** for better visualization of prediction certainty
* **Minimal dependencies** to reduce bundle size and loading times

#### Model & Performance
* **DistilBERT selection**: 40% smaller than BERT while retaining 97% of performance
* **Model caching**: Loaded once at startup to minimize inference latency
* **Volume mounting**: Model directory survives container rebuilds for persistent fine-tuning
* **Deterministic training**: Random seeds set (`random`, `numpy`, `torch`, `transformers`) for reproducible CPU runs


## 6. Deployment Options

### Local Development
```bash
# Backend only
cd backend && python app.py

# Frontend only
cd frontend && open index.html
```


### Environment Variables
```bash
# Model configuration
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
MODEL_PATH=/app/model  # Custom model path

# Server configuration
PORT=8000
HOST=0.0.0.0
WORKERS=1

# CORS settings
ALLOWED_ORIGINS=["http://localhost:3000", "https://yourapp.com"]

# Logging
LOG_LEVEL=INFO
```

### Production Considerations
- **Model versioning**: Tag and version your fine-tuned models
- **Monitoring**: Add application performance monitoring (APM)
- **Caching**: Implement Redis for frequent predictions
- **Rate limiting**: Protect against API abuse
- **HTTPS**: Enable SSL/TLS for production deployments
- **Secrets management**: Use environment variables for sensitive data

---

## 7. Development & Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run smoke tests
pytest -q

# Run with coverage
pytest --cov=backend tests/
```

### Development Workflow
```bash
# Start backend in development mode
cd backend
python app.py --reload

# Frontend development (if using build tools)
cd frontend
# Since we use CDN React, just open index.html
# For production: consider switching to Create React App
```

### Code Quality
```bash
# Format code
black backend/
isort backend/

# Lint
flake8 backend/
pylint backend/

# Type checking
mypy backend/
```

## 8. Extending / Optional Enhancements

* Switch to **GraphQL** using `strawberry-fastapi`.
* Add async request batching with `torchserve` or `ray`.
* Quantise model via `bitsandbytes` or export to ONNX.
* Add GitHub Actions workflow for CI/CD & automated Docker builds.
* Hot-reload weights by watching `backend/model/` with `watchdog`.

---

## 7. License

MIT. 