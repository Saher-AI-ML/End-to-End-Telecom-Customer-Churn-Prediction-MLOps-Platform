# One image that can run BOTH the API and the dashboard.
# docker-compose.yml decides which command each container starts.
FROM python:3.13-slim

# Print logs immediately and don't write .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install the libraries first. Docker caches this layer, so changing your code later
# does not reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what is needed to RUN the project (not the notebook, data or training script)
COPY api/ api/
COPY app/ app/
COPY models/ models/

EXPOSE 8000 8501

# Default command: start the API. docker-compose overrides this for the dashboard.
# --host 0.0.0.0 is required, otherwise the API cannot be reached from outside the container.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
