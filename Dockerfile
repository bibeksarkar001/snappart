# SnapPart backend image: the API and the worker share it (compose picks the command). Built from backend/environment.docker.yml.
# Not yet built in this repo (no Docker on the dev machine); docs/ops.md lists what to check on first build.
FROM mambaorg/micromamba:2.0.5

COPY --chown=$MAMBA_USER:$MAMBA_USER backend/environment.docker.yml /tmp/env.yml
RUN micromamba install -y -n base -f /tmp/env.yml && micromamba clean --all --yes

WORKDIR /srv/backend
COPY --chown=$MAMBA_USER:$MAMBA_USER backend/app ./app
COPY --chown=$MAMBA_USER:$MAMBA_USER backend/capture ./capture
COPY --chown=$MAMBA_USER:$MAMBA_USER backend/cad ./cad
COPY --chown=$MAMBA_USER:$MAMBA_USER backend/measure ./measure

# Non-root (the base image's mambauser). Data lives on a volume; the root filesystem can be mounted read-only.
USER $MAMBA_USER
ENV SNAPPART_DATA_DIR=/data PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--limit-concurrency", "2000", \
     "--timeout-keep-alive", "5", "--h11-max-incomplete-event-size", "16384", "--no-server-header"]
