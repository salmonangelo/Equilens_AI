FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run uses PORT env var
ENV PORT=8000
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV APP_ENV=production

# Run the backend server
CMD ["python", "main.py", "--mode", "backend", "--no-reload"]
