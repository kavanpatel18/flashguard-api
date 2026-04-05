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
                    # ── 11. Run with Gunicorn ─────────────────────────────────────────────────────
# 1 worker, 2 threads — safe for a TF model (models are loaded once per
# worker via the module-level _load cache) within 512MB RAM constraints on Render.
# Timeout set to 120s to allow model cold-start on first inference.
# Bind to PORT environment variable provided by Render, defaulting to 5000.
CMD sh -c "gunicorn --workers 1 --threads 2 --bind 0.0.0.0:${PORT:-5000} --timeout 120 --access-logfile - --error-logfile - --log-level info api_server:app"

                    
