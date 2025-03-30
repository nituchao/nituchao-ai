# 新建 builder 实例
# Docker 默认的 builder 不支持同时指定多个架构，所以要新建一个：
docker buildx create --use --name m1_builder

# 查看并启动 builder 实例：
docker buildx inspect --bootstrap

# 使用 buildx构建：
docker buildx build \
  --platform linux/amd64,linux/arm64
  --push -t prinsss/google-analytics-hit-counter .

# 其中 -t参数指定远程仓库，--push表示将构建好的镜像推送到 Docker 仓库。如果不想直接推送，也可以改成--load，即将构建结果加载到镜像列表中。

# --platform参数就是要构建的目标平台，这里我就选了本机的arm64 和服务器用的 amd64。最后的.（构建路径）注意不要忘了加。

# docker buildx build --platform linux/amd64 --push -t comflag/nituchao-typecho:v1.0.0 ./ 
docker buildx build --platform linux/amd64 --load -t nituchao/devbox-amd64:v1.0.0 ./ 

# 导出镜像
docker save -o image-comflag-nituchao-typecho-amd64-v1.0.0.tar comflag/nituchao-typecho-amd64:v1.0.0