FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-render.txt .

# Install Python dependencies with no cache to reduce image size
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-render.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run the Flask app with Gunicorn in production
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} app:app"]
