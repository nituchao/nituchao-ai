FROM php:8.3-fpm

ARG NITUCHAO_TYPECHO_VERSION=1.0.0

RUN apt -y update && \
    apt -y --no-install-recommends install \
    nginx \
    lsof \
    sqlite3

# Copy the start.sh
RUN mkdir -p /opt/conf
COPY ["archive/start.sh", "/opt/conf/"]

RUN \
    chown -R root:root /opt/conf/ && \
    chmod +x /opt/conf/start.sh

# Expose the port for access
EXPOSE 80/tcp

# Run the servers
CMD ["/bin/bash", "/opt/conf/start.sh"]