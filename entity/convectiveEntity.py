# 对流云相关的实体
from pydantic import BaseModel

class ConvectiveTrackingEntity(BaseModel):
    time:str
    interval:int
    algorithm:str