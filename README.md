# Sentiment Analysis Microservice

This project provides an end-to-end, container-ready solution for **binary sentiment analysis**.
It contains:

1. **Python FastAPI backend** – loads a Hugging Face transformer and exposes a REST endpoint for inference.
2. **Fine-tuning CLI** – retrain the model on your labelled data and hot-swap the weights.
3. **Minimal React frontend** – simple web page to query the API.
4. **Docker Compose** – one-command local deployment on CPU-only machines.

---

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

## 3. API

### `POST /predict`

Request body:
```json
{
  "text": "I absolutely loved it!"
}
```

Successful response (HTTP 200):
```json
{
  "label": "positive",
  "score": 0.987
}
```

Errors return non-200 with `{"detail": <message>}`.

Interactive docs: http://localhost:8000/docs (Swagger UI)

---

## 4. Fine-tuning

Fine-tuning is **optional**. Provide a small JSONL dataset with one record per line:

```json
{"text": "Great product!", "label": "positive"}
{"text": "Worst experience.", "label": "negative"}
```

Run on CPU:

```bash
python finetune.py --data data.jsonl --epochs 3 --lr 2e-5 \
                   --lr_scheduler_type cosine --warmup_steps 500
```

Weights are saved to `backend/model/`. On the next API restart (`docker-compose up`) the service will automatically load the new weights.

Approx. training times (Intel i7-1260P CPU, 3-epoch, 100 samples): **~70 s**.

GPU fine-tune: change the base image to a CUDA variant and add `--device cuda` inside `finetune.py` (not included in default stack).

---

## 5. Design Decisions

* **FastAPI** chosen for its speed, type hints & automatic docs.
* **Transformers pipeline** abstracts preprocessing + postprocessing.
* **DistilBERT SST-2** serves as a robust default English sentiment model.
* Model directory is volume-mounted so weights survive container rebuilds.
* Frontend kept intentionally minimal (CDN React + Nginx) to reduce image sizes.
* Random seeds set (`random`, `numpy`, `torch`, `transformers`) for deterministic CPU runs.

## 🛠 Development

### Run Tests
```bash
# optional smoke tests
pytest -q
```

---

## 6. Extending / Optional Enhancements

* Switch to **GraphQL** using `strawberry-fastapi`.
* Add async request batching with `torchserve` or `ray`.
* Quantise model via `bitsandbytes` or export to ONNX.
* Add GitHub Actions workflow for CI/CD & automated Docker builds.
* Hot-reload weights by watching `backend/model/` with `watchdog`.

---

## 7. License

MIT. 