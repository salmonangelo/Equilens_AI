# EquiLens AI

**AI-Powered Fairness & Bias Auditing Platform**

EquiLens AI is a modular system that evaluates machine learning models for fairness, detects bias across protected attributes, and provides actionable remediation insights through LLM-powered explanations.

---

## Architecture Overview

```
equilens-ai/
├── fairness_engine/     # Core bias metrics & evaluation logic
│   ├── metrics.py       # Statistical fairness metrics (SPD, DI, EOD, etc.)
│   ├── evaluator.py     # Model evaluation orchestrator
│   └── detectors.py     # Bias detection strategies
│
├── mcp_server/          # MCP-compliant local FastAPI server
│   ├── server.py        # FastAPI application & lifespan
│   ├── routes.py        # MCP tool/resource endpoints
│   └── handlers.py      # Request handlers & tool dispatch
│
├── backend/             # Cloud Run production API
│   ├── app.py           # FastAPI application factory
│   ├── routes/          # API route modules
│   │   ├── health.py    # Health & readiness probes
│   │   └── analysis.py  # Fairness analysis endpoints
│   └── models/          # Pydantic schemas
│       └── schemas.py   # Request/response models
│
├── frontend/            # Frontend application (placeholder)
│   └── placeholder.py   # Stub for future UI integration
│
├── prompts/             # LLM prompt templates
│   ├── templates.py     # Jinja2/string prompt builders
│   └── system.py        # System prompt definitions
│
├── config/              # Configuration & environment
│   └── settings.py      # Pydantic Settings for env vars
│
├── tests/               # Test suite
│   ├── test_fairness.py
│   ├── test_backend.py
│   └── test_mcp.py
│
├── main.py              # Application entrypoint
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── .gitignore           # Git ignore rules
```

---

## Prerequisites

- **Python** 3.10+
- **pip** (or **uv** for faster installs)
- A Google Cloud project (for Cloud Run deployment)
- Gemini API key or Vertex AI credentials

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-org/equilens-ai.git
cd equilens-ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env with your actual values
```

### 5. Run the Backend API locally

```bash
python main.py
# or explicitly:
python main.py --mode backend
```

The API will start on **http://localhost:8080**

### 6. Access the API

**API Documentation** (interactive):
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

---

## Local Development Guide

### Running the Backend Locally

#### Step 1: Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

#### Step 2: Configure `.env` for Local Development

Edit `.env` with:

```env
# Local development settings
APP_ENV=local
DEBUG=true
LOG_LEVEL=INFO
BACKEND_PORT=8080

# Optional: Gemini API key (leave empty to skip LLM explanations)
GEMINI_API_KEY=your-api-key-here
```

#### Step 3: Start the Server

```bash
# Start with automatic reload (watches file changes)
python main.py

# Or: disable auto-reload for production-like testing
python main.py --no-reload
```

You should see:

```
================================================================================
🚀 Starting EquiLens AI
================================================================================
Environment: local
Debug Mode: True
Log Level: INFO
Mode: backend
Server will run at http://0.0.0.0:8080
📖 API Documentation available at http://localhost:8080/docs
================================================================================
```

#### Step 4: Test the API

**Option A: Using cURL**

```bash
# 1. Upload a CSV file
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@sample_data.csv"

# Response (copy the dataset_id):
{
  "dataset_id": "abc123def456",
  "filename": "sample_data.csv",
  "rows": 1000,
  "columns": 5,
  "column_names": ["gender", "age", "approved", "predicted", "income"],
  "preview": [...]
}

# 2. Run analysis on the dataset
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123def456",
    "protected_attributes": ["gender"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "loan_approval_v2",
    "skip_anonymization": false
  }'

# Response:
{
  "status": "success",
  "dataset_id": "abc123def456",
  "model_name": "loan_approval_v2",
  "metrics": {
    "gender": {
      "disparate_impact_ratio": 0.72,
      "demographic_parity_difference": -0.15,
      "equal_opportunity_difference": -0.18
    }
  },
  "overall_fair": false,
  "summary": "..."
}
```

**Option B: Using Python**

```python
import requests

BASE_URL = "http://localhost:8080"

# 1. Upload CSV
with open("sample_data.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/v1/upload",
        files={"file": f}
    )
    dataset_id = response.json()["dataset_id"]
    print(f"Dataset uploaded: {dataset_id}")

# 2. Run analysis
analysis_request = {
    "dataset_id": dataset_id,
    "protected_attributes": ["gender"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "loan_approval_v2",
}

response = requests.post(
    f"{BASE_URL}/api/v1/analyze",
    json=analysis_request
)

result = response.json()
print(f"Fair: {result.get('overall_fair')}")
print(f"Metrics: {result.get('metrics')}")
```

**Option C: Using the Interactive Docs**

1. Go to http://localhost:8080/docs
2. Click on `/api/v1/upload` → "Try it out"
3. Upload a CSV file
4. Copy the `dataset_id` from the response
5. Click on `/api/v1/analyze` → "Try it out"
6. Paste the request JSON and execute

### Sample Data

Create `sample_data.csv`:

```csv
gender,age,approved,predicted,income
M,35,1,1,75000
M,42,1,1,85000
F,28,0,0,55000
F,31,1,0,60000
M,45,1,1,95000
F,26,0,0,45000
M,38,1,1,70000
F,33,1,1,72000
M,50,1,1,120000
F,29,0,1,58000
```

### Logging Output

The API logs key events at `INFO` level:

```
[2026-04-25 14:30:00] backend.routes.upload - INFO - 📤 Uploading file: sample_data.csv
[2026-04-25 14:30:00] backend.routes.upload - INFO - ✓ File read successfully (245 bytes)
[2026-04-25 14:30:00] backend.routes.upload - INFO - ✓ CSV parsed: 10 rows, 5 columns
[2026-04-25 14:30:00] backend.routes.upload - INFO - ✓ Dataset stored: abc123def456
[2026-04-25 14:30:00] backend.routes.upload - INFO - ✓ Upload complete: abc123def456
[2026-04-25 14:30:01] backend.routes.analysis - INFO - 📊 Analysis requested: dataset=abc123def456, model=loan_approval_v2
[2026-04-25 14:30:01] backend.routes.analysis - INFO - ✓ Dataset loaded: 10 rows, 5 columns
[2026-04-25 14:30:01] backend.routes.analysis - INFO - ✓ All required columns present
[2026-04-25 14:30:01] backend.routes.analysis - INFO - → Starting fairness metrics computation...
[2026-04-25 14:30:01] backend.routes.analysis - INFO - ✓ Metrics computation complete
[2026-04-25 14:30:01] backend.routes.analysis - INFO - ✓ Analysis complete for model: loan_approval_v2
```

### Error Handling Examples

#### Missing Column

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123def456",
    "protected_attributes": ["nonexistent_column"],
    "target_column": "approved",
    ...
  }'

# Response (422 Unprocessable Entity):
{
  "detail": {
    "error": "Missing required columns",
    "missing_columns": ["nonexistent_column"],
    "available_columns": ["gender", "age", "approved", "predicted", "income"]
  }
}
```

#### Empty File

```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@empty.csv"

# Response (400 Bad Request):
{
  "detail": "Uploaded file is empty."
}
```

#### File Too Large

```bash
# Response (413 Payload Too Large):
{
  "detail": "File exceeds the 10 MB limit."
}
```

#### Dataset Not Found

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "nonexistent", ...}'

# Response (404 Not Found):
{
  "detail": "Dataset 'nonexistent' not found. Upload a CSV first via POST /upload."
}
```

### Health Checks

```bash
# Liveness probe
curl http://localhost:8080/health
# {"status": "healthy", "service": "EquiLens AI", "environment": "local", "timestamp": "2026-04-25T..."}

# Readiness probe
curl http://localhost:8080/ready
# {"status": "ready", "checks": {}}
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| **Port 8080 already in use** | Change port: `BACKEND_PORT=9000 python main.py` |
| **Gemini API key error** | Leave `GEMINI_API_KEY` empty for local testing; LLM explanations will be skipped |
| **"Module not found" error** | Run `pip install -r requirements.txt` |
| **CSV parse error** | Ensure CSV is UTF-8 encoded and has a header row |
| **Memory error with large files** | Max file size is 10 MB; split large datasets |

---

## API Endpoints

### Upload Dataset

```
POST /api/v1/upload

Request:
  - file: CSV file (multipart/form-data)

Response (201 Created):
  {
    "dataset_id": "string",
    "filename": "string",
    "rows": integer,
    "columns": integer,
    "column_names": [string],
    "preview": [object]
  }
```

### Run Analysis

```
POST /api/v1/analyze

Request:
  {
    "dataset_id": "string",
    "protected_attributes": [string],
    "target_column": "string",
    "prediction_column": "string or null",
    "model_name": "string",
    "skip_anonymization": boolean
  }

Response (200 OK):
  {
    "status": "success",
    "dataset_id": "string",
    "model_name": "string",
    "metrics": {object},
    "overall_fair": boolean,
    "summary": "string"
  }
```

### Health Check

```
GET /health

Response (200 OK):
  {
    "status": "healthy",
    "service": "EquiLens AI",
    "environment": "local",
    "timestamp": "2026-04-25T14:30:00.000Z"
  }
```

---

## Development Tips

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov=fairness_engine

# Run specific test file
pytest tests/test_backend.py -v

# Run prompt system validation
pytest tests/test_prompt_system.py -v
```

### Code Style & Linting

```bash
# Format with ruff
ruff format .

# Lint with ruff
ruff check .

# Type check with mypy
mypy backend fairness_engine
```

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG python main.py
```

---

## Next Steps

1. **Prepare Sample Datasets**: Create CSV files with protected attributes and predictions
2. **Test the API**: Use curl or Python requests to test endpoints
3. **Review Logs**: Check console output for INFO/WARNING/ERROR messages
4. **Integrate Gemini**: Set `GEMINI_API_KEY` to enable LLM explanations
5. **Deploy to Cloud**: Follow deployment guide for Google Cloud Run

---

## Additional Resources

- **API Documentation**: http://localhost:8080/docs
- **Prompt Engineering Guide**: `docs/PROMPT_ENGINEERING_GUIDE.md`
- **Quick Reference**: `docs/PROMPT_QUICK_REFERENCE.md`
- **Architecture Overview**: See section below

---
# Server starts at http://localhost:8000
```

### 6. Run the Cloud Run backend (local preview)

```bash
python main.py --mode backend
# API starts at http://localhost:8080
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Module Details

### `fairness_engine`
Core computational module implementing statistical fairness metrics:
- **Statistical Parity Difference (SPD)**
- **Disparate Impact (DI)**
- **Equal Opportunity Difference (EOD)**
- **Predictive Parity**
- Custom metric extensibility via base classes

### `mcp_server`
A local FastAPI server implementing the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) specification, enabling LLM tools to invoke fairness analysis as callable tools.

### `backend`
Production-grade FastAPI application designed for Google Cloud Run deployment. Exposes RESTful endpoints for fairness analysis, model evaluation, and report generation.

### `prompts`
Centralized LLM prompt management with versioned templates for bias explanation, remediation suggestions, and report narratives.

### `config`
Environment-aware configuration using Pydantic Settings with validation, type coercion, and `.env` file support.

---

## Deployment

### Cloud Run

```bash
gcloud run deploy equilens-ai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
#   E q u i l e n s _ A I  
 