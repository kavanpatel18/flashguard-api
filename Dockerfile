# Use Python 3.11 slim as the base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
        PIP_NO_CACHE_DIR=1 \
            PORT=5000

            # Set the working directory
            WORKDIR /app

            # Install system dependencies
            RUN apt-get update && apt-get install -y --no-install-recommends \
                curl \
                    && rm -rf /var/lib/apt/lists/*

                    # Copy requirements file
                    COPY requirements.txt .

                    # Install Python dependencies
                    
                    RUN pip install --no-cache-dir -r requirements.txt

                    # Copy the rest of the application code
                    COPY . .

                    # Expose the port
                    EXPOSE $PORT

                    # Define the command to run the application
                    CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "api_server:app"]

                    
