# 对流云相关的实体
from pydantic import BaseModel
from typing import List
from entity.coordinate import Coordinate
class ConvectiveTrackingEntity(BaseModel):
    time:str
    interval:int
    algorithm:str
    poolScale:int=0