FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY environment.py tasks.py inference.py openenv.yaml README.md /app/

RUN pip install --no-cache-dir openai==1.77.0 pydantic==1.10.15 PyYAML==6.0.2

CMD ["python", "inference.py"]
