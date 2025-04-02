# 区域相关操作的请求体
from pydantic import BaseModel
from typing import List
from entity.coordinate import Coordinate
class RegionData:
    id:str
    points:List[object]
class RegionCoverageEntity(BaseModel):
    time:str
    interval:int=6
    dataType:int=12
    algorithm:str
    thresholds:List[float]
    regions:List[object]