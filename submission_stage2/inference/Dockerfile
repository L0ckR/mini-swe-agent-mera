FROM node:22.23.5-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS pi-runtime
WORKDIR /opt/mera-pi
COPY pi/package.json pi/package-lock.json ./
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund
COPY pi/ ./
RUN node --check worker.mjs

FROM vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3

ENV HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FLASHINFER_CUDA_ARCH_LIST=12.0a \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

WORKDIR /opt/swe-mera
COPY --from=pi-runtime /usr/local/bin/node /usr/local/bin/node
COPY --from=pi-runtime /opt/mera-pi /opt/mera-pi
RUN node --version
COPY model.json serve.py download_model.py ./
# No runtime network access is needed. Downloading at build time also makes
# the exact weights available to all three independent GPU replicas.
RUN python3 download_model.py
COPY supervisor.py pi_service.py ./

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
ENTRYPOINT ["python3", "/opt/swe-mera/supervisor.py"]
