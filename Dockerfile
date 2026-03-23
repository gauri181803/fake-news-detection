FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyG dependencies first (order matters)
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch-geometric==2.5.3
RUN pip install --no-cache-dir \
    torch-scatter==2.1.2 torch-sparse==0.6.18 \
    --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model checkpoints go in /app/models/
# Mount or COPY your .pth files here before building:
#   COPY models/ /app/models/

ENV MODEL_DIR=/app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
