FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with wheel preference
RUN pip install --upgrade pip setuptools wheel
RUN pip install --only-binary=all --prefer-binary -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 7860

# Run with uvicorn - important for proper web server handling
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--timeout-keep-alive", "60"]