FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY backend /app/backend
COPY data /app/data

RUN mkdir -p /app/output

ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/output
ENV ROUTE_RANDOM_STARTS=10
ENV ROUTE_TWO_OPT_ROUNDS=140

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
