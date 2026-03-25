FROM python:3.10-slim

# Accept RUN_ID as build argument
ARG RUN_ID
ENV RUN_ID=${RUN_ID}

WORKDIR /app

# Install dependencies + DVC
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc[s3]

# Create model directory
RUN mkdir -p /app/model

# Mock download of model (simulate fetching from MLflow)
RUN echo "Mock download for model with RUN_ID: ${RUN_ID}" && \
    echo "Model downloaded successfully for RUN_ID: ${RUN_ID}" > /app/model/download_info.txt

# Copy your code
COPY . .

# Run container (example: check model exists)
CMD ["python", "-c", "print('Model ready in /app/model'); import os; print('Contents:', os.listdir('/app/model'))"]