# AWS Lambda Container Image — TensorFlow CPU
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc g++ make libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV LAMBDA_TASK_ROOT=/var/task
WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG CACHE_BUST=6
COPY app.py .
COPY lambda_handler.py .
COPY templates/ templates/
COPY static/ static/
COPY model/ model/

ENTRYPOINT [ "python", "-m", "awslambdaric" ]
CMD [ "lambda_handler.handler" ]
