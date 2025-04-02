import datetime
import numpy as np
from geopy.distance import geodesic,great_circle
from geographiclib.geodesic import Geodesic as Radius
def getTrend(values):
    '''
    计算数组的相关性，用于判断数值的变化趋势
    '''
    values=np.array(values)
    r=np.corrcoef(range(len(values)),values)[0,1]
    if r>0.8:
        return 1
    elif r<-0.8:
        return -1
    else:
        return 0    
def betweenHours(dt1,dt2):
    # 计算两个 datetime 对象之间的差值
    delta = dt2 - dt1
    # 将 timedelta 对象的总秒数转换为小时
    hours = delta.total_seconds() / 3600
    return hours
def getDistance(start_lon,start_lat,end_lon,end_lat):
    '''
    将经纬度变为距离
    '''
    # 定义两个经纬度点
    point1 = (start_lat, start_lon)  
    point2 = (end_lat, end_lon)  

    # 计算距离（单位：公里）
    distance = geodesic(point1, point2).kilometers
    return distance
def getDirection(start_lon,start_lat,end_lon,end_lat):
    '''
    计算两个点的方位角,正北方向作为0度，顺时针递增
    '''
    # 点1基准方位角
    # 方位角是从某点的指北方向线起，依顺时针方向到目标方向线之间的水平夹角
        
    # 计算距离（单位：公里）
    direction=Radius.WGS84.Inverse(start_lat, start_lon,end_lat, end_lon)
    az = direction['azi1'] 
    return az if az>=0 else az+360
def isCandidateEntity(time,start_time,end_time):
    '''
    判断时间是否满足要求
    '''
    if start_time<=time and time<=end_time:
        return True
    return False
