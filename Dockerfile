# AI Academic Chatbot - FastAPI app
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY personalization_helper.py .
COPY personalization_module.py .
COPY resume_analyzer.py .
COPY shared_database.py .
COPY setup.py .

# Optional: copy shared_data if needed at runtime (create dir if not copying)
RUN mkdir -p /app/shared_data

# Expose FastAPI port
EXPOSE 8000

# Run with 0.0.0.0 so the server is reachable from outside the container
# No --reload in production for stability
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
