# ── AWS Lambda Container Image ──────────────────────────────────────────────
# Using python:3.11-slim as base (has gcc/build tools available via apt)
# then installing the Lambda Runtime Interface Client (awslambdaric)
FROM python:3.11-slim

# Install build tools (needed for numpy, h5py compilation)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Lambda function directory
ENV LAMBDA_TASK_ROOT=/var/task
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache-bust arg — change this value to force re-copy of app files only
ARG CACHE_BUST=3
COPY app.py .
COPY lambda_handler.py .
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

# Lambda Runtime Interface Client entry point
ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "lambda_handler.handler" ]
