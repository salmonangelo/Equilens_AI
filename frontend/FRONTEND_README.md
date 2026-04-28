# EquiLens AI — Frontend UI

A clean, minimal HTML/JS frontend for the EquiLens fairness analysis platform.

## Quick Start

### Option 1: Python Built-in Server
```bash
cd frontend
python -m http.server 5000
```

Open browser: `http://localhost:5000`

### Option 2: Node.js (with npx)
```bash
cd frontend
npx serve -l 5000
```

### Option 3: VS Code Live Server
Right-click `index.html` → "Open with Live Server"

---

## Features

- **Upload CSV**: Select a dataset for analysis
- **Configure Analysis**: Specify protected attributes, target column, prediction column
- **Run Fairness Audit**: Call backend API to compute metrics
- **View Results**: Display metrics, risk scores, and AI explanation (if Gemini available)

## Configuration

Edit `script.js` to change backend URL:

```javascript
const API_BASE_URL = 'http://localhost:8080';  // Your backend URL
```

## API Endpoints Used

- `POST /api/v1/upload` - Upload CSV dataset
- `POST /api/v1/analyze` - Run fairness analysis
- `GET /api/health` - Health check

## Architecture

- **index.html** - Markup + styled components
- **script.js** - Client-side logic (upload, analyze, display results)

No frameworks—pure HTML/CSS/JS for minimal dependencies.

## Privacy

All data handling:
- CSV uploads go directly to backend
- No data stored in browser localStorage
- All anonymization happens server-side
- Metrics only sent to Gemini (no raw data)

## Styling

- Dark theme (purple/cyan accent)
- Responsive design
- Smooth animations
- Real-time error/success messages
- Loading states

---

For backend setup, see [../FRONTEND_QUICKSTART.md](../FRONTEND_QUICKSTART.md)
