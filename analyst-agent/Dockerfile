# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY tools.py .
COPY __init__.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose the port (Cloud Run uses 8080)
EXPOSE 8080

# Use gunicorn for production deployment
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
