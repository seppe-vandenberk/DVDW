# Lightweight simulator/viewer: all model operations are remote HTTP calls.
FROM python:3.12-slim AS simulator
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements-http.txt .
RUN python -m pip install --no-cache-dir -r requirements-http.txt
COPY connect4 ./connect4
EXPOSE 8080
CMD ["python", "-m", "uvicorn", "connect4.simulation_service:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# Official PyTorch image includes the CUDA runtime; only the host driver is needed.
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime AS api
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_DIR=/app/models DQN_DEVICE=cpu
# The base image uses externally managed system Python. Install HTTP packages in
# a virtual environment that reuses its pip and CUDA-enabled PyTorch. The base
# image omits ensurepip, so use the existing pip through python -m pip.
RUN python -m venv --system-site-packages --without-pip /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
COPY requirements-http.txt .
RUN python -m pip install --no-cache-dir -r requirements-http.txt
COPY connect4 ./connect4
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "connect4.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
