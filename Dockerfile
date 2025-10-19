FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

ENV MODEL_PATH="Qwen/Qwen3-8B"
ENV LLM_SERVER_URL="http://localhost:8005"

# Run the application
CMD ["python", "main.py"]
