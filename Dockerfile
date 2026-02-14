# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (some OA libs might need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local openoa source
COPY openoa/ ./openoa/
COPY pyproject.toml .

# Install the local package in editable mode (or just ensure it's in path)
RUN pip install -e .

# Copy application files
COPY main.py .
COPY index.html .

# Copy example data (needed for the demo calculation)
# Note: Ensure this directory is populated with at least the La Haute Borne CSVs
COPY examples/data/la_haute_borne/ ./examples/data/la_haute_borne/

# Expose port
EXPOSE 8080

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
