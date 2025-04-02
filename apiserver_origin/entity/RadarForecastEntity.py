# 对流预报
from pydantic import BaseModel
from typing import List
# 单点预报
class RadarPointForecastEntity(BaseModel):
    # 算法名字
    code:str
    time:str
    points:List[object]
# 区域预报
class RadarRegionForecastEntity(BaseModel):
    code:str
    time:str
    areas:List[object]