# 🚀 EquiLens AI - Quick Start Guide

A complete fairness analysis platform with FastAPI backend + minimal HTML/JS frontend.

## Setup & Run (Local Development)

### Step 1: Start the Backend

Open a terminal and run:

```bash
# Default: runs on http://localhost:8080
python main.py

# Or, specify a different port
BACKEND_PORT=8000 python main.py
```

You'll see output like:
```
✓ Starting EquiLens AI Backend
Environment: development
Server will run at http://0.0.0.0:8080
📖 API Documentation available at http://localhost:8080/docs
```

### Step 2: Start the Frontend

Open a **new terminal** (keep backend running) and run:

```bash
cd frontend

# Option A: Python's built-in server
python -m http.server 5000

# Option B: Using npm/Node.js
npx serve -l 5000

# Option C: VS Code Live Server
# Right-click index.html → "Open with Live Server"
```

Then open your browser:
```
http://localhost:5000
```

### Step 3: Try the Demo

1. **Upload CSV**: Select any CSV file with your data
2. **Configure columns**:
   - Protected Attributes: `gender, race` (comma-separated)
   - Target Column: `approved` (or your label column)
   - Prediction Column: `predicted` (optional)
   - Model Name: `loan_approval_v2` (optional)
3. **Click "1. Upload CSV"** → Wait for confirmation
4. **Click "2. Run Analysis"** → Wait for results
5. **View Results**: Metrics, risk scores, and AI insights

## Troubleshooting

### Backend Port Mismatch
If backend runs on different port, edit `frontend/script.js` line 2:
```javascript
const API_BASE_URL = 'http://localhost:YOUR_PORT';
```

### CORS Errors
**Error:** `Access to XMLHttpRequest blocked by CORS policy`

**Solution:** Backend already includes CORS for `localhost:5000`. If using a different port:
1. Stop the backend
2. Edit `config/settings.py` line 40:
   ```python
   CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5000,http://YOUR_PORT"
   ```
3. Restart backend

### "Cannot connect to backend"
- [ ] Backend is running (`python main.py`)
- [ ] Check port matches API_BASE_URL in `frontend/script.js`
- [ ] Try: `curl http://localhost:8080/docs` in another terminal
- [ ] Check firewall/antivirus isn't blocking ports

### Upload Fails
- [ ] CSV file is less than 10 MB
- [ ] CSV is valid (no encoding issues)
- [ ] Column names don't have spaces (or match exactly)

### Results Not Showing
1. Open browser console: **F12** → **Console** tab
2. Look for error messages
3. Check **Network** tab to see API responses
4. Verify column names match your CSV exactly

## File Structure

```
solution_challenge/
├── main.py                 # Start backend: python main.py
├── backend/               
│   ├── app.py            # FastAPI app + CORS config
│   ├── routes/           # /upload, /analyze endpoints
│   └── services/         # Fairness metrics, analysis
├── config/
│   └── settings.py       # Configuration (ports, CORS, etc.)
└── frontend/             # ← Frontend files
    ├── index.html        # Main page
    ├── script.js         # API calls, state management
    ├── SETUP.md          # Detailed frontend guide
    └── ...
```

## API Endpoints Used

### POST /upload
Uploads a CSV file for analysis.

```bash
curl -X POST http://localhost:8080/upload \
  -F "file=@data.csv"
```

### POST /analyze
Runs fairness analysis on the uploaded dataset.

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "abc123",
    "protected_attributes": ["gender", "race"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "model_v1",
    "skip_anonymization": false
  }'
```

**Response includes:**
- Fairness metrics per protected attribute
- Risk scores (LOW/MEDIUM/HIGH)
- AI-generated explanation
- Compliance information

## Features

✅ Upload CSV files  
✅ Configure analysis parameters  
✅ View fairness metrics (Disparate Impact, Demographic Parity, etc.)  
✅ Risk scoring with visual indicators  
✅ AI-powered explanations (via Gemini)  
✅ Clean, responsive UI  
✅ Loading states & error handling  
✅ No heavy frameworks (vanilla HTML/JS)  

## Next Steps

- [ ] Test with sample CSV data
- [ ] Customize CORS origins for your network
- [ ] Add more columns to your CSV
- [ ] View API docs: `http://localhost:8080/docs`
- [ ] Deploy to cloud (see Docker Compose or deploy instructions)

## Testing Sample Data

Create `test_data.csv`:
```csv
id,gender,race,approved,predicted
1,M,White,1,1
2,F,Black,0,1
3,M,White,1,1
4,F,Black,1,0
5,M,Asian,1,1
6,F,Hispanic,0,0
```

Upload it and analyze with:
- Protected Attributes: `gender, race`
- Target Column: `approved`
- Prediction Column: `predicted`

## Support

**View Logs:**
- Backend logs: Check terminal where `python main.py` is running
- Frontend logs: Browser DevTools (F12 → Console)
- API errors: Network tab (F12 → Network)

**Check API Health:**
```bash
curl http://localhost:8080/health
```

## Environment Variables

Set before running `python main.py`:

```bash
# Backend port
export BACKEND_PORT=9000

# Frontend will fetch from
API_BASE_URL = 'http://localhost:9000'

# Debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Gemini API (optional, for LLM explanations)
export GEMINI_API_KEY="your-key-here"
```

## Docker (Optional)

```bash
docker-compose up
```

This runs both backend (8080) and frontend (5000) in containers.

---

**Ready?** Start the backend, then the frontend. Happy analyzing! 🎉
