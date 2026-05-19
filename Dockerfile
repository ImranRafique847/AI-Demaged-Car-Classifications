# ── AWS Lambda Container Image ──────────────────────────────
# Base image: AWS Lambda Python 3.11 runtime
FROM public.ecr.aws/lambda/python:3.11

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install CPU-only dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY lambda_handler.py .
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

# Lambda handler entry point
CMD ["lambda_handler.handler"]
