# The improver network's only way out: an HTTPS proxy that lets the model APIs through and nothing else.
FROM public.ecr.aws/docker/library/debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends tinyproxy \
    && rm -rf /var/lib/apt/lists/*
COPY tinyproxy.conf allowlist /etc/tinyproxy/
USER nobody
EXPOSE 8888
CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
