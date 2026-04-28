# EquiLens Frontend - Setup & Run Guide

A minimal, clean HTML + JavaScript frontend for fairness analysis.

## Quick Start

### Prerequisites
- **Backend running**: FastAPI server on `http://localhost:8080` (default) or `http://localhost:8000` (if you override)
- **Browser**: Any modern browser (Chrome, Firefox, Safari, Edge)
- **Python 3.8+** (for running local server)

### Option 1: Using Python's Built-in Server (Recommended for Development)

```bash
# Navigate to the frontend directory
cd frontend

# Start a simple HTTP server on port 5000
python -m http.server 5000
```

Then open your browser and go to:
```
http://localhost:5000
```

### Option 2: Using Node.js (if you have it installed)

```bash
cd frontend

# Install serve globally (one-time)
npm install -g serve

# Start server on port 5000
serve -l 5000
```

Then open: `http://localhost:5000`

### Option 3: Using Live Server (VS Code Extension)

1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html` → "Open with Live Server"
3. Automatically opens in your browser

## How to Use

1. **Upload CSV**
   - Click "Upload CSV" button
   - Select your dataset file
   - Fill in:
     - **Protected Attributes**: Comma-separated list (e.g., `gender, race`)
     - **Target Column**: The binary label column (e.g., `approved`)
     - **Prediction Column** (optional): Model predictions (e.g., `predicted`)
     - **Model Name** (optional): Human-readable ID (e.g., `loan_approval_v2`)
   - Click "1. Upload CSV"

2. **Run Analysis**
   - Once file is uploaded, click "2. Run Analysis"
   - Wait for results (usually 5-30 seconds)

3. **View Results**
   - Fairness metrics by protected attribute
   - Disparate Impact, Demographic Parity, Equal Opportunity ratios
   - Fairness Risk Scores (LOW/MEDIUM/HIGH) per attribute
   - AI-generated insights and recommendations

## API Endpoints Used

The frontend calls two endpoints:

### POST /upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data.csv"
```

**Response:**
```json
{
  "dataset_id": "a1b2c3d4",
  "rows": 1000,
  "columns": 15,
  "uploaded_at": "2026-04-25T10:30:00Z"
}
```

### POST /analyze
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "a1b2c3d4",
    "protected_attributes": ["gender", "race"],
    "target_column": "approved",
    "prediction_column": "predicted",
    "model_name": "loan_approval_v2",
    "skip_anonymization": false
  }'
```

## File Structure

```
frontend/
├── index.html       # Main HTML (form, results display, styles)
├── script.js        # JavaScript (API calls, state management, DOM updates)
└── SETUP.md         # This file
```

## Features

✅ **Upload CSV files** to backend  
✅ **Configure analysis parameters** easily  
✅ **Display fairness metrics** in clean cards  
✅ **Risk scoring** with color-coded levels (LOW/MEDIUM/HIGH)  
✅ **AI explanations** from LLM analysis  
✅ **Loading states** for async operations  
✅ **Error handling** with user-friendly messages  
✅ **Responsive design** (works on mobile too)  
✅ **No framework dependencies** (vanilla HTML/JS)  

## Troubleshooting

### "Cannot connect to localhost:8000"
- Ensure your FastAPI backend is running
- Check that the backend is on port 8000
- Try: `curl http://localhost:8000/docs` to verify

### CORS errors in browser console
Add CORS middleware to your FastAPI app:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### File upload fails
- Check file size (max 10 MB)
- Ensure CSV is valid and readable
- Try with sample data first

### Results not showing
- Check browser console for errors (F12 → Console tab)
- Verify column names match your CSV headers exactly
- Ensure protected attributes and target column exist in your data

## Customization

**Change API URL:**
Edit line 4 in `script.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this
```

**Change port:**
Replace `5000` with your preferred port in the server command.

**Styling:**
Edit the `<style>` block in `index.html` to customize colors, fonts, and layout.

## Next Steps

- Add form validation improvements
- Implement CSV preview before upload
- Add export results as PDF/JSON
- Connect to authentication system
- Deploy to cloud (Vercel, Netlify, AWS, etc.)

## Support

For issues, check:
1. Browser console (F12 → Console)
2. Backend logs (FastAPI terminal)
3. Network tab (F12 → Network) to see API calls
