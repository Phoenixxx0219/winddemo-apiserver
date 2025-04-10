import os
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from functools import lru_cache

from entity.convectiveEntity import ConvectiveTrackingEntity
from entity.trackingPointEntity import TrackingPointEntity
from function import convectiveTracking, find_earliest_entity

# 创建应用程序，app是应用程序名
app = FastAPI()  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许的域名列表
    allow_credentials=True,
    allow_methods=["*"],           # 允许所有 HTTP 方法
    allow_headers=["*"],           # 允许所有请求头
)

# 静态目录挂载（访问 /static/ 路径时会从 ./static 目录读取文件）
app.mount("/static", StaticFiles(directory="static"), name="static")

# 统一返回对象 
class Result:
    def __init__(self):
        pass
    def ok(self,data):
        return {
            "code":200,
            "message":"success",
            "data":data
        }
    def error(self,message):
        return {
            "code":500,
            "message":f"{message}",
            "data":""
        }

# 缓存封装：构造一个 key（元组），避免直接缓存 dict（不可哈希）
@lru_cache(maxsize=128)
def cached_convective_tracking(startTime: str, algorithm: str, interval: int):
    data = {
        "startTime": startTime,
        "algorithm": algorithm,
        "interval": interval
    }
    return convectiveTracking(data)

# 处理 OPTIONS 预检请求（可选调试方案）
@app.options("/api/convective/tracking")
async def options_tracking(request: Request):
    return {}

# API请求

# 获取真实图片数据
@app.get("/realimage/{date}/{type}/real/{filename}")
async def get_real_image(date: str, type: str, filename: str):
    file_path = f"static/ImageData/{date}/{type}/real/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

# 获取预测图片数据
@app.get("/forcastimage/{date}/{type}/forcast/{time}/{filename}")
async def get_forcast_image(date: str, type: str, time: str, filename: str):
    file_path = f"static/ImageData/{date}/{type}/forcast/{time}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

# 获取单体轮廓数据
@app.post("/api/convective/tracking")
async def getConvectiveTracking(item: ConvectiveTrackingEntity):
    print(f"开始请求 tracking，请求参数: {item}")
    try:
        loop = asyncio.get_event_loop()
        responseData = await loop.run_in_executor(
            None,
            cached_convective_tracking,
            item.time,
            item.algorithm,
            item.interval
        )
        if responseData is None:
            return Result().error(f"对流追踪请求失败, error: 缺少数据 {item.time}")
        return Result().ok(responseData)
    except Exception as e:
        return Result().error(f"对流追踪请求失败, error: {e}")

# 获取单点查询信息
@app.post("/api/convective/point")
async def getTrackingEntityByPoint(item: TrackingPointEntity):
    print(f"请求包含坐标点的对流单体追踪, 参数: {item}")
    try:
        loop = asyncio.get_event_loop()
        responseData = await loop.run_in_executor(
            None,
            cached_convective_tracking,
            item.time,
            item.algorithm,
            item.interval
        )
        if responseData is None:
            return Result().error(f"对流追踪请求失败, error: 缺少数据 {item.time}")
        target_point = (item.lat, item.lon)
        earliest = find_earliest_entity(responseData, target_point)
        return Result().ok(earliest)
    except Exception as e:
        return Result().error(f"对流追踪请求失败, error: {e}")

# 启动服务
if __name__ == '__main__':
    uvicorn.run("apiServer:app", port=8080, reload=True)
