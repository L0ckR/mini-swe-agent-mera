FROM vllm/vllm-openai:v0.31.0

ENV HF_HOME=/opt/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FLASHINFER_CUDA_ARCH_LIST=12.0a \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

WORKDIR /opt/swe-mera
COPY model.json serve.py download_model.py ./
# No runtime network access is needed. Downloading at build time also makes
# the exact weights available to all three independent GPU replicas.
RUN python3 download_model.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000
ENTRYPOINT ["python3", "/opt/swe-mera/serve.py"]
