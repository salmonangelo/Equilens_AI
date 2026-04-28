# EquiLens AI — Local Runtime Setup

Complete guide to running EquiLens AI locally without cloud deployment.

---

## Prerequisites

- **Python** 3.10+ (verify: `python --version`)
- **pip** (verify: `pip --version`)
- Virtual environment (optional but recommended)

### Optional
- **Gemini API Key** (for AI explanations) — get from [Google AI Studio](https://aistudio.google.com)
- **Ollama** (for local Gemma schema analysis) — download from [ollama.ai](https://ollama.ai)

---

## Step 1: Setup Python Environment

### Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Step 2: Configure Environment Variables

### Copy Template
```bash
cp .env.example .env
```

### Edit `.env`
```bash
# .env

# Application
APP_ENV=local
APP_NAME=EquiLens AI
DEBUG=true
LOG_LEVEL=INFO

# API
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8080
CORS_ORIGINS=http://localhost:5000,http://localhost:3000

# Optional: Gemini API Key (for AI explanations)
GEMINI_API_KEY=

# Optional: Gemma/Ollama (for local schema analysis)
OLLAMA_BASE_URL=http://localhost:11434
```

> **Note**: GEMINI_API_KEY is optional. System works without it (uses fallback responses).

---

## Step 3: Start Backend API

### Command
```bash
python main.py
```

### Expected Output
```
================================================================================
✓ Starting EquiLens AI Backend
  Environment: local
  Debug: True
  CORS Origins: http://localhost:5000
================================================================================
INFO: Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

### Verify
- API Docs: http://localhost:8080/docs
- Health: http://localhost:8080/api/health

---

## Step 4: Start Frontend (Optional)

### In a NEW terminal (keep backend running):

```bash
cd frontend

# Option A: Python
python -m http.server 5000

# Option B: Node.js (if you have npx)
npx serve -l 5000

# Option C: VS Code
# Right-click index.html → "Open with Live Server"
```

### Access
- Frontend: http://localhost:5000
- Upload CSV → Configure → Analyze → View Results

---

## Step 5: Run Demo Script (Local Analysis)

### In a NEW terminal:

```bash
# No Gemini (local metrics only)
python analyze_demo.py --no-explanation

# With Gemini (requires GEMINI_API_KEY)
GEMINI_API_KEY=your-key-here python analyze_demo.py

# Custom dataset
python analyze_demo.py --dataset /path/to/data.csv

# Save output to file
python analyze_demo.py --output results.json
```

### Output
- Metrics: DI, DPD, EOD, FRS computed locally
- Risk Score: HIGH/MEDIUM/LOW classification
- Privacy Summary: Shows what data was/wasn't sent to APIs
- JSON Report: Full analysis result

---

## Optional: Setup Gemma (Local LLM)

### Install Ollama
1. Download from https://ollama.ai
2. Install and run

### Pull Gemma Model
```bash
ollama pull gemma:7b
ollama serve  # Starts on http://localhost:11434
```

### Verify
```bash
curl http://localhost:11434/api/tags
```

### Use in EquiLens
- Backend auto-detects Ollama
- Used for column classification (PII / protected / target)
- Gracefully degrades if unavailable
- Schema analysis only (NOT fairness metrics)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  LOCAL RUNTIME STACK                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Frontend (http://localhost:5000)                          │
│  ├─ Upload CSV → /api/v1/upload                           │
│  └─ Analyze → /api/v1/analyze                             │
│         ↓                                                  │
│  Backend (http://localhost:8080)                          │
│  ├─ Routes: health, upload, analyze                       │
│  ├─ Services:                                             │
│  │  ├─ analysis_service.py (metrics)                      │
│  │  ├─ explanation_service_v2.py (Gemini)                │
│  │  ├─ gemma_integration.py (Ollama)                      │
│  │  └─ full_analysis_pipeline.py (orchestrator)           │
│  ├─ Privacy:                                              │
│  │  └─ privacy/validator.py (blocks raw data)             │
│  └─ Engines:                                              │
│     ├─ fairness_engine/metrics.py (DI, DPD, EOD)          │
│     ├─ fairness_engine/scoring.py (FRS)                   │
│     └─ fairness_engine/anonymizer.py (PII)               │
│         ↓                                                  │
│  Optional: Gemini API (explanation)                       │
│  Optional: Ollama (local Gemma)                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | development | local \| development \| staging \| production |
| `APP_NAME` | EquiLens AI | Application name |
| `DEBUG` | true | Enable debug logging |
| `LOG_LEVEL` | INFO | DEBUG \| INFO \| WARNING \| ERROR |
| `BACKEND_HOST` | 0.0.0.0 | Server bind address |
| `BACKEND_PORT` | 8080 | Server port |
| `CORS_ORIGINS` | localhost:5000 | Comma-separated CORS origins |
| `GEMINI_API_KEY` | (unset) | Google Gemini API key (optional) |

---

## Running Tests

### All Tests
```bash
python -m pytest tests/ -v
```

### Specific Module
```bash
python -m pytest tests/test_fairness.py -v
python -m pytest tests/test_privacy_validation.py -v
```

### With Coverage
```bash
python -m pytest tests/ --cov=backend --cov=fairness_engine -v
```

---

## Troubleshooting

### Backend won't start
```bash
# Check if port 8080 is in use
lsof -i :8080    # macOS/Linux
netstat -aon | findstr :8080  # Windows

# Use different port
BACKEND_PORT=9000 python main.py
```

### CORS errors
Edit `.env`:
```bash
CORS_ORIGINS=http://localhost:5000,http://localhost:3000
```

### Gemini API errors
- Verify GEMINI_API_KEY is set
- Check quota at https://aistudio.google.com
- System uses fallback responses if API fails (graceful degradation)

### Ollama not found
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- System auto-detects and falls back to heuristics if unavailable

---

## 60-Second Demo

### Terminal 1: Backend
```bash
python main.py
# Runs on http://localhost:8080
```

### Terminal 2: Demo Script
```bash
python analyze_demo.py --no-explanation
# Outputs: JSON report with metrics, risk score, privacy summary
```

### Terminal 3 (Optional): Frontend
```bash
cd frontend && python -m http.server 5000
# Open http://localhost:5000 in browser
```

---

## Production Deployment

For Cloud Run / Kubernetes:

```bash
# Build Docker image
docker build -t equilens-ai .

# Deploy to Cloud Run
gcloud run deploy equilens-ai \
  --image equilens-ai \
  --platform managed \
  --region us-central1 \
  --set-env-vars GEMINI_API_KEY=xxx
```

See `Dockerfile` for details.

---

## Support

- **API Docs**: http://localhost:8080/docs
- **README**: [./README.md](./README.md)
- **Quick Start**: [./QUICKSTART_LOCAL.md](./docs/QUICKSTART_LOCAL.md)
