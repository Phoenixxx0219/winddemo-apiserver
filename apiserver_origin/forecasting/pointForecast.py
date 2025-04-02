# 点预警和区域预警函数
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import datetime
from shapely.geometry import Point,Polygon
from traffic import trafficPoint
from function import convectiveTracking
import math
from forecasting.common import *
import config
import asyncio
from aiofile import async_open
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
from cache.cache import random_expire,generateUUID,getLock,releaseLock
import random
import logging
logger=logging.getLogger('apiServer')
# 在pointForecasting函数前添加哈希生成函数
def generate_point_hash(data):
    import hashlib
    hash_str = f"{data['code']}_{data['time']}"
    for point in sorted(data['points'], key=lambda x: x['id']):
        hash_str += f"|{point['id']}"
        hash_str += f"@{point['leadTime']}"
        hash_str += f"#{point['minEchoIntensity']}"
        hash_str += f"({point['coordinates']['lat']:.4f},{point['coordinates']['lng']:.4f})"
    
    combined = f"{hashlib.md5(hash_str.encode()).hexdigest()}"
    return combined

def isPointInPolygon(lon,lat,outline):
    '''
    判断点是否在多边形内
    '''
    # 把纬经度变为经纬度
    outline_new=[(lon,lat) for lat,lon in outline]
    # 构建多边形
    poly=Polygon(outline_new)
    # 构建点
    point=Point(lon,lat)
    return point.within(poly)
def createPoint(point_date,point,entity,query_time,entity_direction):
    '''
    构造点预测的返回值
    {
            id: "Point1",
            distance: '100', // 距离，单位：千米
            direction: 70, // 移动方向，单位：度
            speed: '50', // 速度，单位：千米/小时
            trend: 0, // 趋势， 1: 增强 0：保持 -1：减弱
            impact: {
                startTime: '2023-07-01 08:00',
                endTime: '2023-07-01 08:18',
        }
    }
    '''
    id,coor,index,minEchoIntensity=point
    lon,lat=coor['lng'],coor['lat']
    spanData=entity['spanData']
    # 当前时刻对流单体所在span，如果当前时刻对流单体未产生，那么使用对流单体产生时的时刻作为代替
    cur_span=spanData[0]
    available_span_list=[]
    # 找到结束影响的时间
    for span in spanData:
        span_time=span['time']
        span_time=datetime.datetime.strptime(span_time,'%Y-%m-%d %H:%M:%S')
        if span_time==query_time:
            cur_span=span
        # 如果单体包含点，并且平均强度大于阈值，则该单体仍然在影响当前点
        if point_date<=span_time and isPointInPolygon(lon,lat,span['outline']) and span['avgValue']>=minEchoIntensity:
            available_span_list.append(span)
        elif point_date<span_time:
            # 后面不可能找到结束时间了
            break
    
    # 计算距离，当前时刻对流单体还要走多远才能影响point
    dis=getDistance(cur_span['lon'],cur_span['lat'],available_span_list[0]['lon'],available_span_list[0]['lat'])
    # 计算角度，对流单体的移动角度,直接用奇凤的结果
    direction=entity_direction
    # 计算速度，对流单体的移动速度
    speed_time=betweenHours(query_time,point_date)
    speed=dis/speed_time
    # 计算对流单体强度趋势，使用影响区间内的相关性进行计算
    values=[span['avgValue'] for span in available_span_list]
    trend=getTrend(values)
    impact={
        "startTime":available_span_list[0]['time'],
        "endTime":available_span_list[-1]['time']
    }
    # 构造返回点
    point_res={
        "id":id,
        "distance":dis,
        "direction":direction,
        "speed":speed,
        "trend":trend,
        "impact":impact
    }
    return point_res

def pointAvailableSpanData(point_date,point,spanData):
    '''
    判断当前对流是否影响point
    '''
    id,coor,index,minEchoIntensity=point
    lon,lat=coor['lng'],coor['lat']
    for span in spanData:
        span_time=span['time']
        span_time=datetime.datetime.strptime(span_time,'%Y-%m-%d %H:%M:%S')
        # 找到point_date相同的span，看看span是否包含了点
        if point_date==span_time:
            return  isPointInPolygon(lon,lat,span['outline'])
    return False
async def pointForecasting(data):
    '''
    点的雷达预警，影响力从未来开始计算
    请求参数:
    {
    code: "difftrans_deploy_3h",
    time: "2024-11-05 08:06",
    points: [
        {
            id: "Point1",
            coordinates: { lat: 39.904211, lng: 116.407395 },
            leadTime: 10,
            minEchoIntensity: 25
        },
        {
            id: "Point2",
            coordinates: { lat: 39.904211, lng: 116.407395 },
            leadTime: 10,
            minEchoIntensity: 25
        }
    ]
    }
    返回数据:
    {
    points: [
        {
            id: "Point1",
            distance: '100', // 距离，单位：千米
            direction: 70, // 移动方向，单位：度
            speed: '50', // 速度，单位：千米/小时
            trend: 0, // 趋势， 1: 增强 0：保持 -1：减弱
            impact: {
                startTime: '2024-11-05 08:12',
                endTime: '2024-11-05 08:18',
            }
        }
    ]
    }
    '''
    # 寻找第一个满足阈值需求的红绿灯点
    def findFirstIndex(threshold,traffic_list):
        for i in range(len(traffic_list)):
            if traffic_list[i]>=threshold:
                return i
        return -1
    # CPU密集型计算多线程处理
    def process_point(point):
        id,coor,index,minEchoIntensity=point
        if index>=0:
            flag=False
            point_date=date+datetime.timedelta(minutes=6*(index+1))
            candidate_entities=[]
            
            # 找到候选单体
            for entity in entities:
                start_time=entity["startTime"]
                end_time=entity["endTime"]
                start_time=datetime.datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S')
                end_time=datetime.datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S')
                if isCandidateEntity(point_date,start_time,end_time):
                    candidate_entities.append(entity)
            
            # 找到第一个包含point的候选单体作为影响的对流云
            for entity in candidate_entities:
                spanData=entity['spanData']
                if pointAvailableSpanData(point_date,point,spanData):
                    # 当前entity包含点，由于entity是按时间顺序升序，因此直接返回
                    return_point=createPoint(point_date,point,entity,date,entity['direction'])
                    return return_point      
        return None
    
    date=datetime.datetime.strptime(data['time'],"%Y-%m-%d %H:%M")
    date=datetime.datetime(year=date.year,month=date.month,day=date.day,hour=date.hour,minute=date.minute,second=0)
    # 读取redis
    redis_key=config.GLOBAL_CONFIG['RADAR_POINT_KEY']+date.strftime("%Y%m%d%H%M")+":"+generate_point_hash(data)
    redisWorker=config.GLOBAL_CONFIG['REDIS_CLIENT']
    value=redisWorker.getJSON(redis_key)
    if value is not None:
        return value
    token=generateUUID()+"Z"+str(random.randint(0,100000))
    lock=getLock(redis_key,token)
    try:
        if lock:
            logger.info(f"获取pointForecasting缓存失败，获取锁成功{date}:{redis_key}:{token}")
            # 双重检查缓存
            value = redisWorker.getJSON(redis_key)
            if value is not None:
                logger.info(f"获取pointForecasting缓存失败，双重检查成功{date}:{redis_key}:{token}")
                return value
            
            # 请求所有点的红绿灯
            traffic_coor=[]
            for point in data['points']:
                coordinates=point['coordinates']
                traffic_coor.append([coordinates['lng'],coordinates['lat']])
            traffic_date=date.strftime("%Y%m%d%H%M")
            traffic_list=await trafficPoint(traffic_coor,traffic_date,data['code'])
            # 根据红绿灯过滤请求
            i=0
            filter_point_list=[]
            for point in data['points']:
                id=point['id']
                coor=point['coordinates']
                leadTime=point['leadTime']
                minEchoIntensity=point['minEchoIntensity']
                # 将leadTime转为列表中的index,可能的雷达预警结果下标为[10,10+cnt),影响力从未来时刻算起
                cnt=math.ceil(leadTime/6)
                # 寻找第一个影响到该点的雷达时刻
                index=findFirstIndex(minEchoIntensity,traffic_list[i][10:10+cnt])
                # 记录每个point受到的影响
                filter_point_list.append([id,coor,index,minEchoIntensity])
                i+=1
            # 请求对流云的追踪结果
            tracking_query={}
            tracking_query["startTime"]=data["time"]
            tracking_query["interval"]=6
            tracking_query["algorithm"]=data["code"]
            tracking_query["poolScale"]=0
            tracking_res=await asyncio.get_event_loop().run_in_executor(config.GLOBAL_CONFIG["CPU_POOLS"],convectiveTracking,tracking_query)
            entities=tracking_res["entities"]
            # 遍历对流追踪的结果，找到第一个影响point的单体
            res={}
            res["points"]=[]    
            # 多线程处理
            tasks=[asyncio.get_event_loop().run_in_executor(config.GLOBAL_CONFIG['CPU_POOLS'],process_point,point) for point in filter_point_list]
            return_points=await asyncio.gather(*tasks)
            # 过滤掉None
            return_points=[point for point in return_points if point is not None]
            res["points"]=return_points
            # 写入redis
            redisWorker.setJSON(redis_key,res,ex=random_expire(3,10)*60)
            return res
        else:
            logger.info(f"获取pointForecasting缓存失败，获取锁失败{date}:{redis_key}:{token}")
            await asyncio.sleep(random.uniform(0.01,0.1))
            return await pointForecasting(data)
    finally:
        if lock:
            logger.info(f"释放pointForecasting锁{date}:{redis_key}:{token}")
            releaseLock(redis_key,token)
async def main():
    import time
    st=time.time()
    # 初始化key
    config.set_config("TRACKING_KEY","tracking:")
    config.set_config("TRAFFIC_KEY","traffic:")
    config.set_config("RADAR_POINT_KEY",'radar_point_forecasting:')
    config.set_config("RADAR_REGFION_KEY",'radar_region_forecasting:')
    config.set_config("COVERAGE_KEY",'coverage:')
    # 初始化config
    config.set_config("REDISHOST","localhost")
    config.set_config("REDISPASS","ices123456")
    config.set_config("REDISPORT",6379)
    import redis
    from cache.cache import RedisWorker
    # 初始化redis连接
    pool=redis.ConnectionPool(host="localhost", port=6379, password="ices123456", db=0,max_connections=1000)
    redis_client = RedisWorker(pool)
    config.set_config("REDIS_CLIENT", redis_client)
    # 初始化计算的线程池或者进程池
    executors=ThreadPoolExecutor(max_workers=os.cpu_count()*4)
    config.set_config("CPU_POOLS",executors)
    data= {
        "code": "radar_difftrans_deploy_3h",
        "time": "2024-06-12 05:00",
        "points": [
            {
                "id": "连胜围",
                "coordinates": { 
                    "lat": 22.23,
                    "lng": 113.47 },
                "leadTime": 180,
                "minEchoIntensity": 25
            },
            {
                "id": "观澜",
                "coordinates": { 
                    "lat": 22.71,
                    "lng": 114.035 },
                "leadTime": 180,
                "minEchoIntensity": 25
            },
            {
                "id": "南朗",
                "coordinates": { 
                    "lat": 22.53,
                    "lng": 113.56 },
                "leadTime": 180,
                "minEchoIntensity": 25
            },
        ]
    }
    res=await pointForecasting(data)
    print(res)
    ed=time.time()
    print(f"{(ed-st)*1000}ms")
if __name__=='__main__':
    asyncio.run(main())