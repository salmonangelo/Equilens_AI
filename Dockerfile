FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Cloud Run uses PORT env var
ENV PORT=8080

# Run the backend server
CMD ["python", "main.py", "--mode", "backend"]
