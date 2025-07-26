# Use Python 3.11 runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-cloudrun.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-cloudrun.txt

# Copy application code
COPY . .

# Set environment variables for Cloud Run
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run the application with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 cloud_run_deployment:app
