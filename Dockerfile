FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY personalization_helper.py .
COPY personalization_module.py .
COPY resume_analyzer.py .
COPY shared_database.py .
COPY setup.py .

RUN mkdir -p /app/shared_data

# Copy supervisor config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose all ports
EXPOSE 8000 8001 8002

CMD ["/usr/bin/supervisord"]
