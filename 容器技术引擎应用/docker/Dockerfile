# dockerfile: ubuntu-dev.dockerfile
# FROM --platform=linux/amd64 ubuntu:24.04
FROM ubuntu:24.04
LABEL Caster="zhangliang605@gmail.com"

# timezone
ARG TZ=Asia/Shanghai
ENV TZ=${TZ}

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER root

# 更换阿里云源，在国内可以加快速度
RUN sed -i "s/security.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
    sed -i "s/archive.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list && \
    sed -i "s/security-cdn.ubuntu.com/mirrors.aliyun.com/" /etc/apt/sources.list
RUN  apt-get clean

# 更新源，安装相应工具
RUN apt-get update && apt-get install -y \
    zsh \
    vim \
    sudo \
    wget \
    curl \
    python3 \
    git-core \
    openssh-server

#  安装 zsh，以后进入容器中时，更加方便地使用 shell
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh \
    && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && sed -i 's/^plugins=(/plugins=(zsh-autosuggestions zsh-syntax-highlighting z /' ~/.zshrc \
    && chsh -s /bin/zsh

# 创建 nituchao 用户
RUN useradd --create-home --no-log-init --shell /bin/zsh -G sudo nituchao
RUN adduser nituchao sudo
RUN echo 'nituchao:rootroot' | chpasswd

# 为 nituchao 安装 omz
USER nituchao
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh \
    && cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting \
    && sed -i 's/^plugins=(/plugins=(zsh-autosuggestions zsh-syntax-highlighting z /' ~/.zshrc

# 删除 apt/lists，可以减少最终镜像大小
USER root
RUN rm -rf /var/lib/apt/lists/*

# ssh public key for host client ssh login without passowrd
USER nituchao
ARG HOST_SSH_PUB_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCW49fw7yFGR3q2SOnK4MAzKS8ZxM8kqVy9K1onMn2ssjXAoDT17dOJ3yTQdjG2PQ837qqZ8v9lJd5y8or3tEQzbhcx9+uF8ZDyWAIP3FkqQDcX2uiWB/D2h4J+9LjekJbewmICKEsDeBQFgjE3bHZbBtnyBNOJlTI6NdeMCOzMbwT5od0JQuwUESTBCuRr+4CzScE7UtcnaHa/5hYh+AQqQ+kTeaJ4WgnvnaoRTJE2g5HH3/X+Wj5UMiOWXfFpO7zACbJ0GeLICe/CNvTk3+L8mcDF89sHniAS+ogIBNqWQBeezVNTjZSal2mE+5AEJd0jFvjmA8gkf7Yr1Fzc35pNf7ydsT1G6IKxBki8Ib4n3irBWJnIYHl5+CkfzjThSnl8Pn0tIu3D6SoW/ywVEGhkmJZ7bjkEvs3mvbYGyuNJ6XwlXfZaZRSrN607ztvmYlhXum1PrHFSwCCHo4HZ7Hn3bpV+K+pOLxtrUJZMjGbYt4PeuZ0Vpe13oFV1CCr5aeVakxWjhniUqm38c6uRYsUNxxujGPZgns1X0hcA3l3iF1iZyf/PQKcExUwbY3AiS8jI/sEh395wql3erGF7Y506EfwIO1T8dIS9/CZVX1clS2dLd656e3l7lq167xjXDZSLktu1BRFCiMLKaWl6Xn5epjCT81zRy62IHOiQ7jPoOQ== zhangliang.thanks@bytedance.com"

RUN mkdir -p /home/nituchao/.ssh && touch /home/nituchao/.ssh/authorized_keys && echo $HOST_SSH_PUB_KEY >> /home/nituchao/.ssh/authorized_keys
RUN mkdir -p /home/nituchao/codes
RUN mkdir -p /home/nituchao/data

USER root
RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
RUN ssh-keygen -A

RUN echo 'root:rootroot' | chpasswd

RUN service ssh start

EXPOSE 22

CMD ["/usr/sbin/sshd","-D"]