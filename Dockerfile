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
EXPOSE 7860

# Run the Gradio app directly
CMD ["python", "app.py"]
