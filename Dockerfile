FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu --no-deps "torch==2.14.1+cpu" \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY evaluation/chunk_manifest.json evaluation/retrieval_dataset.json evaluation/relevance_judgments.json evaluation/multi_query_queries.json ./evaluation/

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /tmp/huggingface \
    && chown -R app:app /app /tmp/huggingface
USER app

ENTRYPOINT ["python"]
CMD ["scripts/ask.py", "--help"]
