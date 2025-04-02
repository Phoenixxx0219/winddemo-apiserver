# GFS API使用说明

{
  "time": "2024-06-04 00:00",
  "interval": 6,
  "algorithm": "radar_difftrans_deploy_3h"
}

## 源代码使用说明
1. 安装环境
```
pip install -r requirements.txt
```
2. 安装Redis
3. 运行`apiServer.py`,注意传递参数
```
python apiServer.py --app_host 0.0.0.0 --app_port web服务端口 --redis_host redis主机 --redis_port redis端口 --redis_pw redis的密码
```
## 可执行文件使用说明
```
# 打包
conda activate apiServer
pyinstaller --onefile apiServer.py
# 执行
cd ./dist
./apiServer --app_host 0.0.0.0 --app_port web服务端口 --redis_host redis主机 --redis_port redis端口 --redis_pw redis的密码
```
## docker使用说明
### 创建docker的网络连接
```
sudo docker network create my_network
```
### 启动redis docker
```
# 下载docker
sudo docker pull redis
# 启动redis的docker
docker run --restart always -d \
--network my_network \
--name redis-container \
--hostname redis-container \
-e REDIS_PASSWORD=ices123456 \
-p 6379:6379 \
redis:latest \
redis-server --requirepass ices123456 --maxmemory 2gb
```
### 启动pyhon服务的docker
1. 打包pyhon服务docker镜像（如果已经打包好则跳过此步）
```
sudo docker build -f Dockerfile -t apiserver-app-image .
```
2. 启动pyhon服务的docker
```
# 后台运行
sudo docker run --env-file .env --restart always -d \
--network  my_network \
-v /mnt/disk/caddy/:/data/ \
-v ./log/:/app/log/ \
-p 9000:9000 \
--name apiserver-app-docker  apiserver-app-image
```
3. docker的注意事项
```
--network my_network:将docker容器放入my_network的网络中
/data/:docker数据目录
/log/:docker的日志目录
9000:docker默认端口和--app_port一致
--app_host:建议直接填0.0.0.0
--app_port:服务器端口
--redis_host:redis的服务器的新主机名，若redis为相同network下的docker容器，则为容器名
--redis_port:redis数据库的端口，默认为6379
--redis_pw:redis的密码，默认为ices123456
```
## 加密
1. 运行python setup.py build_ext --inplace
2. 将所有同名的py文件删除，so文件会替代py文件，如果py文件和so文件同时存在，那么会优先使用py文件
## 日志
- [X] 20241016实现批量读取功能，未debug
- [X] 20241028完成从本地读取nc数据，最终版
- [X] 20241121完成红绿灯数据读取
- [X] 20241217添加redis缓存优化对流追踪5000ms->263ms
- [X] 20241218添加后台任务定期更新redis
- [X] 20241226使用fatapi重构的请求接口
- [X] 20241231备份好redis的docker
- [X] 20250101备份好python服务器的docker
- [X] 20250103优化docker的打包流程，目前能够在任意主机上一键打包
- [x] 20250104优化后的docker测试通过
- [x] 20250225完成点雷达预警
- [x] 20250227完成区域雷达预警
- [x] 20250228新增线程池，提高了后端的并发能力
- [x] 20250307新增redis完成性能优化
- [x] 20250308新增cython加密功能