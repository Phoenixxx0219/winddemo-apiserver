# 单点查询的实体
from pydantic import BaseModel

class TrackingPointEntity(BaseModel):
    time: str         # 对流追踪的开始时间
    interval: int     # 时间间隔
    algorithm: str    # 使用的算法名称
    lat: float        # 前端传入的纬度（坐标点）
    lon: float        # 前端传入的经度（坐标点）