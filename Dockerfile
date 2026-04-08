FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY environment.py tasks.py inference.py openenv.yaml README.md pyproject.toml /app/
COPY server /app/server/

RUN pip install --no-cache-dir openai 'pydantic>=2.0.0' PyYAML fastapi uvicorn openenv-core

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
