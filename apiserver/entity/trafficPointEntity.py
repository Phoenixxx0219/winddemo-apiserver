# 红绿灯查询的实体类
from pydantic import BaseModel
from typing import List
from entity.coordinate import Coordinate
class SingleAlgorithmTrafficPoint(BaseModel):
    # 请求时间yyyy-MM-dd HH:mm
    time:str
    interval:int=6
    dataType:int=12
    algorithm:str
    points:object
    
class MultiAlgorithmTrafficPoint(BaseModel):
    # 请求时间yyyy-MM-dd HH:mm
    time:str
    interval:int=6
    dataType:int=12
    algorithms:List[str]
    point:object