# Stage 1: train the model from the public dataset, so the image's artefact is reproducible.
FROM python:3.13-slim AS train
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
RUN python -m src.train_export

# Stage 2: slim runtime image with only serving dependencies, running as a non-root user.
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/app/artifacts
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt && useradd --system --uid 10001 app
COPY src/__init__.py src/api.py src/record.py ./src/
COPY --from=train /app/artifacts ./artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=2)"
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
