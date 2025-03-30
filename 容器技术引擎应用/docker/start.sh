# build docker
docker build -t nituchao/devbox-amd64:v1.0.0  ./

# start docker
docker run --rm -d -p 2022:22 \
-v /Users/bytedance/Documents/Workspace/nituchao/nituchao_codes:/home/nituchao/codes \
-v /Users/bytedance/Documents/Workspace/nituchao/nituchao_data:/home/nituchao/data \
--name devbox \
nituchao/devbox-amd64:v1.0.0 

docker run --rm -d -p 2022:22 \
-v /Users/bytedance/Documents/Workspace/nituchao:/home/nituchao \
--name devbox \
nituchao/devbox-amd64:v1.0.0 

# ananconda
cd /opt/workspace
cd /opt/anaconda
wget --user-agent="Mozilla" https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.3.0-Linux-x86_64.sh