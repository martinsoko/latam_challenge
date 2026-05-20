# syntax=docker/dockerfile:1.2
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY challenge/ ./challenge/
COPY data/ ./data/

# Expose the API port
EXPOSE 8080

# Run uvicorn
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
