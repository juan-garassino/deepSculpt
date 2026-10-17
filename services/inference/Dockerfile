FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/inference/requirements.txt /app/services/inference/requirements.txt
RUN pip install --no-cache-dir -r /app/services/inference/requirements.txt

COPY services/inference /app/services/inference

ENV PYTHONPATH=/app

# Run unprivileged: the API only writes to /tmp and $HOME (matplotlib cache).
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8080

CMD ["uvicorn", "services.inference.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
