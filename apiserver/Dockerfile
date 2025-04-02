# 使用官方 Python 3 镜像作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 将项目的 requirements.txt 复制到容器内的工作目录
COPY requirements.txt /app/

# 安装linux的依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get update && apt-get install -y libglib2.0-0
RUN apt-get update && apt-get install -y gcc python3-dev
# 使用清华源安装依赖
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制打包后的可执行文件到容器中
COPY . /app/

# 创建容器的 /data 目录作为挂载点
VOLUME ["/data/","/log/"]

# 暴露容器的 9000 端口
EXPOSE 9000

# # 开机自动运行
ENTRYPOINT ["python","apiServer.py"]