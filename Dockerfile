FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-render.txt requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 7860

# Run the Gradio app directly
CMD ["python", "app.py"]
