# tg-curator: one container running `curator run`, everything it keeps in the /data volume.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# An unprivileged user owns the data; uid/gid 1000 match the first user on most servers, so a
# bind-mounted folder stays readable by its owner.
RUN groupadd --gid 1000 curator \
    && useradd --uid 1000 --gid 1000 --home-dir /data --no-create-home \
       --shell /usr/sbin/nologin curator

COPY pyproject.toml README.md LICENSE /build/
COPY src /build/src
# The postgres extra is included so `TG_CURATOR_DATABASE_URL` works in the container too.
RUN pip install "/build[postgres]" && rm -rf /build

# /data is created here, owned by curator and private, so a new named volume inherits both.
RUN mkdir -m 0700 /data && chown curator:curator /data

# HOME and the curator's home are the volume; the model cache lives there too, so it is
# downloaded once per volume, not once per image. Wait mode keeps a container with incomplete
# settings alive and waiting instead of crash-looping under `restart: unless-stopped`.
ENV HOME=/data \
    TG_CURATOR_HOME=/data \
    TG_CURATOR_WAIT_FOR_SETTINGS=1 \
    HF_HOME=/data/models \
    HF_HUB_DISABLE_TELEMETRY=1

VOLUME ["/data"]
WORKDIR /data
USER curator

# Without `init: true` (a plain `docker run`) curator is PID 1, and the kernel drops a SIGTERM
# that has no handler yet: while the container waits for settings or downloads the model,
# `docker stop` would hang until SIGKILL. Python always handles SIGINT, and once running the
# service shuts down on it exactly as on SIGTERM.
STOPSIGNAL SIGINT

CMD ["curator", "run"]
