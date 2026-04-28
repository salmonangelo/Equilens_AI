# EquiLens AI — Quick Start Guide (Local Development)

**Get the API running in 5 minutes.**

---

## ⚡ Quick Setup

### 1️⃣ Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install packages
pip install -r requirements.txt
```

### 2️⃣ Configure Environment

```bash
# Copy template
cp .env.example .env

# (Optional) Edit .env to set GEMINI_API_KEY if you want LLM features
# For now, leave it empty for local testing
```

### 3️⃣ Start the Server

```bash
python main.py
```

You should see:

```
================================================================================
🚀 Starting EquiLens AI
================================================================================
Environment: development
Debug Mode: True
Log Level: INFO
Mode: backend
Server will run at http://0.0.0.0:8080
📖 API Documentation available at http://localhost:8080/docs
================================================================================
```

### 4️⃣ Test the API

**In a new terminal:**

```bash
# Run the test script
python tests/test_api_local.py
```

Or use curl directly (see examples below).

---

## 🧪 Test Examples

### Using cURL

**Step 1: Create a sample CSV file**

```csv
gender,age,approved,predicted,income
M,35,1,1,75000
F,28,0,0,55000
M,42,1,1,85000
F,31,1,0,60000
```

Save as `test_data.csv`

**Step 2: Upload**

```bash
curl -X POST http://localhost:8080/api/v1/upload \
  -F "file=@test_data.csv"
```

Response:
```json
{
  "dataset_id": "abc123def456",
  "filename": "test_data.csv",
  "rows": 4,
  "columns": 5,
  "column_names": ["gender", "age", "approved", "predicted", "income"],
  "preview": [...]
}
```

**Step 3: Analyze**

```bash
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123def456",
    "protected_attributes": ["gender"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "test_model",
    "skip_anonymization": false
  }'
```

### Using Python

```python
import requests

BASE = "http://localhost:8080"

# Upload
with open("test_data.csv", "rb") as f:
    r = requests.post(f"{BASE}/api/v1/upload", files={"file": f})
    dataset_id = r.json()["dataset_id"]

# Analyze
r = requests.post(f"{BASE}/api/v1/analyze", json={
    "dataset_id": dataset_id,
    "protected_attributes": ["gender"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "test",
})

print(r.json())
```

### Using Interactive Docs

1. Open http://localhost:8080/docs
2. Click on `/api/v1/upload`
3. Click "Try it out" → upload a CSV
4. Copy the `dataset_id`
5. Click on `/api/v1/analyze`
6. Paste the analysis request

---

## 📊 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/upload` | Upload CSV file |
| `POST` | `/api/v1/analyze` | Run fairness analysis |

---

## 🎯 What Happens in Analysis

1. **Validation**: Checks if required columns exist
2. **Anonymization**: Detects and redacts PII (unless skipped)
3. **Metrics**: Computes fairness scores (DI, DPD, EOD)
4. **Risk Score**: Generates overall fairness risk
5. **(Optional) LLM**: Calls Gemini for explanations (if API key set)

---

## 📝 Logging Output

Key log messages (INFO level):

```
✓ File read successfully (245 bytes)
✓ CSV parsed: 10 rows, 5 columns
✓ Dataset stored: abc123def456
✓ Dataset loaded: 10 rows, 5 columns
✓ All required columns present
→ Starting fairness metrics computation...
✓ Metrics computation complete
✓ Analysis complete for model: loan_approval_v2
```

Enable debug logging:
```bash
LOG_LEVEL=DEBUG python main.py
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| **"Cannot connect to localhost:8080"** | Make sure backend is running (`python main.py`) |
| **"Port 8080 already in use"** | `BACKEND_PORT=9000 python main.py` |
| **CSV parse error** | Ensure UTF-8 encoding with header row |
| **Missing columns error** | CSV must have exact column names you specified |
| **Gemini errors** | Leave `GEMINI_API_KEY` empty; local mode works without it |

---

## 📚 Next Steps

1. **Explore the API**: Go to http://localhost:8080/docs
2. **Test with real data**: Upload your own CSV files
3. **Check logs**: Monitor console output for INFO/WARNING/ERROR
4. **Enable Gemini**: Set `GEMINI_API_KEY` for LLM explanations
5. **Read full docs**: See `README.md` for detailed guide

---

## 🔗 Documentation

- **Full Setup Guide**: See `README.md` → "Local Development Guide"
- **Prompt Engineering**: See `docs/PROMPT_ENGINEERING_GUIDE.md`
- **Quick Reference**: See `docs/PROMPT_QUICK_REFERENCE.md`
- **API Docs**: http://localhost:8080/docs (interactive)

---

**Ready to go! 🚀**

Run `python main.py` and enjoy the API!
