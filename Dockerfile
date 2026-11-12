FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN useradd --system --create-home --uid 10001 wastevision
COPY --from=builder /install /usr/local
WORKDIR /app
COPY models ./models
USER wastevision
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"]
CMD ["uvicorn", "wastevision.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
