docker run -d \
-p 9080:80 \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/html:/var/www/html \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/php-fpm.d:/usr/local/etc/php-fpm.d \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/nginx:/etc/nginx/sites-available \
--name typecho_v0 \
comflag/nituchao-typecho:v1.0.0

docker run -d \
-p 9080:80 \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/html:/var/www/html \
--name typecho_v0 \
docker.io/nituchao/typecho:v0




docker run -it \
-p 9080:80 \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/html:/var/www/html \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/php-fpm.d:/usr/local/etc/php-fpm.d \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/nginx:/etc/nginx/sites-available \
--name typecho_v0 \
docker.io/nituchao/typecho:v0 \
/bin/bash

docker run -it \
-p 9080:80 \
-v /Users/bytedance/Documents/Workspace/nituchao_codes/typecho/docker/html:/var/www/html \
--name typecho_v0 \
docker.io/nituchao/typecho:v0 \
/bin/bash