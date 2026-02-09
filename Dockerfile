FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser
USER appuser

ENV JUPYTER_TOKEN=${JUPYTER_TOKEN:-lab}
EXPOSE 8888
CMD ["bash", "-lc", "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=$JUPYTER_TOKEN --NotebookApp.allow_origin='*' --NotebookApp.allow_root=True --NotebookApp.base_url=/"]