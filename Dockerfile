FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mutable state (journal, profile) goes to the mounted volume.
ENV DATA_DIR=/data

CMD ["python", "-m", "velveteentrade", "bot"]
