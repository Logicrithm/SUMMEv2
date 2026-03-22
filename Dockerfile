FROM python:3.11-slim

# Install system dependencies (specifically ffmpeg for audio chunking and yt-dlp)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Flask port
EXPOSE 5000

# Set environment variables for production (disallow local threading clashes)
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK="TRUE"

# Use gunicorn as the production WSGI server instead of Werkzeug
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
