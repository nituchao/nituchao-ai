#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

# Run the php-fpm server
/usr/local/sbin/php-fpm -D -R

# Run the Nginx server
/usr/sbin/nginx -g "daemon off;"