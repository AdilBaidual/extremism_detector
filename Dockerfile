FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    HOST=0.0.0.0 \
    PORT=8000

RUN useradd --create-home appuser
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core ./core
COPY data ./data
COPY app_final.py .

ARG TRAINING_DATA=data/training_data_mock.csv
RUN python -m core.train_model --data "$TRAINING_DATA"

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "app_final.py"]
