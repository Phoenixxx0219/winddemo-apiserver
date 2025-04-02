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
from coverage import coverage_rate
import asyncio
from aiofile import async_open
import config
from cache.cache import getLock,releaseLock,generateUUID,random_expire
import random
import logging
logger=logging.getLogger('apiServer')
def generate_region_hash(data):
    import hashlib
    # 序列化关键参数
    hash_str = f"{data['code']}_{data['time']}"
    for area in sorted(data['areas'], key=lambda x: x['id']):
        hash_str += f"|{area['id']}"
        hash_str += f"@{area['leadTime']}"
        hash_str += f"#{area['minEchoIntensity']}"
        hash_str += f"${area['minEchoCoverage']}"
        for coord in area['coordinates']:
            hash_str += f"({coord['lat']:.4f},{coord['lng']:.4f})"
    return hashlib.md5(hash_str.encode()).hexdigest()
def AreaAndPolygon(coordinates,outline):
    '''
    判断区域和对流云单体的交并比
    '''
    # 把纬经度变为经纬度
    outline_new=[(lon,lat) for lat,lon in outline]
    # 创建多边形
    poly_area = Polygon(coordinates)
    poly_convective = Polygon(outline_new)

    # 计算交集
    intersection = poly_area.intersection(poly_convective)

    # 计算面积
    area_square = poly_area.area
    intersection_square = intersection.area

    # 计算比重
    ratio = 100*(intersection_square / area_square)
    return ratio
def createArea(point_date,area,entity,query_time,entity_direction):
    '''
    构造点预测的返回值
    {
            id: "Area1",
            distance: '100', // 距离，单位：千米
            direction: 70, // 移动方向，单位：度
            speed: '50', // 速度，单位：千米/小时
            trend: 0, // 趋势， 1: 增强 0：保持 -1：减弱
            maxCoverage:覆盖率,
            impact: {
                startTime: '2023-07-01 08:00',
                endTime: '2023-07-01 08:18',
        }
    }
    '''
    id,coor_list,index_list,minEchoIntensity,minEchoCoverage,coverage_rate_list=area
    coordinates=[[coor['lng'],coor['lat']] for coor in coor_list]
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
        if point_date<=span_time and AreaAndPolygon(coordinates,span['outline'])>=minEchoCoverage and span['avgValue']>=minEchoIntensity:
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
    maxCoverage=max([AreaAndPolygon(coordinates,span['outline']) for span in available_span_list])
    impact={
        "startTime":available_span_list[0]['time'],
        "endTime":available_span_list[-1]['time']
    }
    # 构造返回点
    area_res={
        "id":id,
        "distance":dis,
        "direction":direction,
        "speed":speed,
        "trend":trend,
        "maxCoverage":maxCoverage,
        "impact":impact
    }
    return area_res

def areaAvailableSpanData(point_date,area,spanData):
    '''
    判断当前对流是否影响area
    '''
    id,coor_list,index_list,minEchoIntensity,minEchoCoverage,coverage_rate_list=area
    coordinates=[[coor['lng'],coor['lat']] for coor in coor_list]
    for span in spanData:
        span_time=span['time']
        span_time=datetime.datetime.strptime(span_time,'%Y-%m-%d %H:%M:%S')
        # 找到point_date相同的span，看看span的覆盖率和强度是否满足要求
        if point_date==span_time:
            return  span['avgValue']>=minEchoIntensity and AreaAndPolygon(coordinates,span['outline'])>=minEchoCoverage
    return False
async def regionForecasting(data):
    '''
    区域的雷达预警，影响力从未来开始计算
    请求参数:
    {
        "code": "radar_difftrans_deploy_3h",
        "time": "2024-06-12 05:00",
        "areas”: [
            {
                id: "区域1",
                coordinates: [{ lat: 22.5306, lng: 113.5631 }, { lat: 39.904211, lng: 116.407395 }, { lat: 39.904211, lng: 116.407395 }, { lat: 22.5306, lng: 113.5631 }],
            leadTime: 10,
            minEchoIntensity: 25,
            minEchoCoverage: 20
        },
        {
            id: "区域2",
            coordinates: [{ lat: 22.5306, lng: 113.5631 }, { lat: 39.904211, lng: 116.407395 }, { lat: 39.904211, lng: 116.407395 }, { lat: 22.5306, lng: 113.5631 }],
            leadTime: 10,
            minEchoIntensity: 25,
            minEchoCoverage: 20
        }
    ]
    }
    返回数据:
    {
    area: [
        {
            id: "区域",
            distance: '100', // 距离，单位：千米
            direction: 70, // 移动方向，单位：度
            speed: '50', // 速度，单位：千米/小时
            trend: 0, // 趋势， 1: 增强 0：保持 -1：减弱
            maxCoverage:65,
        impact: {
                startTime: '2024-11-05 08:12',
                endTime: '2024-11-05 08:18',
            }
        }
    ]
    }
    '''
    # 寻找所有满足阈值需求的候选时刻
    def findAllIndex(threshold,coverage_rate_list):
        res=[]
        for i in range(len(coverage_rate_list)):
            if coverage_rate_list[i]*100>=threshold:
                res.append(i)
        return res
    async def get_coverage(area):
        coverage_coor = [[coor['lng'], coor['lat']] for coor in area['coordinates']]
        coverage_date=date.strftime("%Y%m%d%H%M")
        return await coverage_rate(coverage_coor, coverage_date, data['code'], area['minEchoIntensity'])
    def process_area(area):
        id,coor_list,index_list,minEchoIntensity,minEchoCoverage,coverage_rate_list=area
        if len(index_list)>=0:
            flag=False
            # 遍历所有可能的候选时间段
            for index in index_list:
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

                # 找到第一个在该时刻满足覆盖率要求的单体
                for entity in candidate_entities:
                    spanData=entity['spanData']
                    if areaAvailableSpanData(point_date,area,spanData):
                        # 当前entity满足要求，由于entity是按时间顺序升序，因此直接返回
                        return_area=createArea(point_date,area,entity,date,entity['direction'])
                        return return_area
        return None
    date=datetime.datetime.strptime(data['time'],"%Y-%m-%d %H:%M")
    date=datetime.datetime(year=date.year,month=date.month,day=date.day,hour=date.hour,minute=date.minute,second=0)
    # 查看redis缓存
    redis_key=config.GLOBAL_CONFIG['RADAR_REGFION_KEY']+date.strftime("%Y%m%d%H%M")+":"+generate_region_hash(data)
    redisWorker=config.GLOBAL_CONFIG['REDIS_CLIENT']
    value=redisWorker.getJSON(redis_key)
    if value is not None:
        return value
    # 缓存未命中，请求数据，使用分布式锁防止高并发
    token=generateUUID()+"Z"+str(random.randint(0,100000))
    lock=getLock(redis_key,token)
    try:
        if lock:
            logger.info(f"获取regionForecasting缓存失败，获取锁成功{date}:{redis_key}:{token}")
            value=redisWorker.getJSON(redis_key)
            if value is not None:
                return value
            
            # 请求所有区域的覆盖率
            coverage_rate_list=[]
            coverage_tasks = [get_coverage(area) for area in data['areas']]
            coverage_rate_list = await asyncio.gather(*coverage_tasks)

            # 根据覆盖率过滤请求
            i=0
            filter_area_list=[]
            for area in data['areas']:
                id=area['id']
                coor_list=area['coordinates']
                leadTime=area['leadTime']
                minEchoIntensity=area['minEchoIntensity']
                minEchoCoverage=area['minEchoCoverage']
                # 将leadTime转为列表中的index,可能的雷达预警结果下标为[10,10+cnt),影响力从未来时刻算起
                cnt=math.ceil(leadTime/6)
                # 寻找所有影响到该区域的雷达时刻
                index_list=findAllIndex(minEchoCoverage,coverage_rate_list[i][10:10+cnt])
                # 记录每个point受到的影响
                filter_area_list.append([id,coor_list,index_list,minEchoIntensity,minEchoCoverage,coverage_rate_list[i][10:10+cnt]])
                i+=1
            # 请求对流云的追踪结果
            tracking_query={}
            tracking_query["startTime"]=data["time"]
            tracking_query["interval"]=6
            tracking_query["algorithm"]=data["code"]
            tracking_query["poolScale"]=0
            tracking_res=await asyncio.get_event_loop().run_in_executor(config.GLOBAL_CONFIG['CPU_POOLS'],convectiveTracking,tracking_query)
            entities=tracking_res["entities"]
            # 多线程计算区域相关信息
            res={}
            res["areas"]=[]
            tasks=[asyncio.get_event_loop().run_in_executor(config.GLOBAL_CONFIG["CPU_POOLS"],process_area,area) for area in filter_area_list]
            return_areas=await asyncio.gather(*tasks)
            # 过滤掉None
            return_areas=[area for area in return_areas if area is not None]
            res["areas"]=return_areas
            redisWorker.setJSON(redis_key,res,ex=random_expire(3,10)*60)
            return res
        else:
            logger.info(f"获取regionForecasting缓存失败，获取锁失败{date}:{redis_key}:{token}")
            await asyncio.sleep(random.uniform(0.01, 0.1))
            return await regionForecasting(data)
    finally:
        if lock:
            logger.info(f"释放regionForecasting锁{date}:{redis_key}:{token}")
            releaseLock(redis_key,token)
def main():
    data= {
        "code": "radar_difftrans_deploy_3h",
        "time": "2024-06-12 05:00",
        "areas": [
            {
                "id": "2区",
                "coordinates": [
                    {"lng": 113.5533, "lat": 21.985},
                    {"lng": 113.69833333333334, "lat": 22.16},
                    {"lng": 113.665, "lat": 22.2283},
                    {"lng": 113.5631, "lat": 22.5306},
                    {"lng": 113.4833, "lat": 22.8817},
                    {"lng": 113.3283333333333, "lat": 22.625},
                    {"lng": 113.5533, "lat": 21.985}
                ],
                "leadTime": 180,
                "minEchoIntensity": 25,
                "minEchoCoverage": 20
            },
            {
                "id": "北一区",
                "coordinates": [
                    {"lng": 113.78878, "lat": 22.66807},
                    {"lng": 113.80547814113521, "lat": 22.676755708700327},
                    {"lng": 113.79915067930631, "lat": 22.8375394211903},
                    {"lng": 113.7146, "lat": 22.795975},
                    {"lng": 113.78878, "lat": 22.66807} 
                ],
                "leadTime": 180,
                "minEchoIntensity": 25,
                "minEchoCoverage": 20
            }
        ]
    }
    res=generate_region_hash(data)
    print()
if __name__=='__main__':
    import time
    import asyncio
    st=time.time()
    main()
    # 初始化key
    config.set_config("TRACKING_KEY","tracking:")
    # 初始化config
    config.set_config("REDISHOST","localhost")
    config.set_config("REDISPASS","ices123456")
    config.set_config("REDISPORT",6379)
    ed=time.time()
    print(f"{(ed-st)*1000}ms")

