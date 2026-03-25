FROM python:3.10-slim

WORKDIR /app

# Install dependencies + DVC
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install dvc[s3]  # change if needed

# Pull the trained model from your repo
RUN mkdir -p /app/model
RUN dvc get https://github.com/RiwanAshraf/mlops-assignment4 model -o /app/model

# Copy your code
COPY . .

# Run container (example: check model exists)
CMD ["python", "-c", "print('Model ready in /app/model'); import os; print(os.listdir('/app/model'))"]