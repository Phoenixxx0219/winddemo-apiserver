# 实际的功能函数
import logging
import datetime
import os
import sys
from traffic import trafficPoint
from coverage import coverage_rate
from tracking.convective import monomer_tracking
from cache.cache import RedisWorker,random_expire,getLock,releaseLock,generateUUID
import config
sys.path.append(os.getcwd())
import logging
import hashlib
import random
import asyncio
logger=logging.getLogger('apiServer')
def satisfyTime(dt,interval:int):
    min=dt.minute
    if min%interval==0:
        return True
    else:
        return False
def satisfyPoolScale(poolScale):
    if poolScale==0 or poolScale== 4 or poolScale==16:
        return True
    else:
        return False
async def trafficPointFromPNG(data):
    '''
    从PNG中查询红绿灯信息
    {   
        //预报时间
        "startTime": "2024-11-04 00:00",
        //时间间隔
        "interval": 6,
        //数据类型
        "type": 0
       //算法编号类型
        "algorithm": "difftrans_deploy_3h",
        //经度数组
        coordinates:[[110,20],[112,32],....]
    }
    '''
    def generate_traffic_png_hash(data):

        # 构造哈希字符串
        hash_str = f"{data['algorithm']}_{data['startTime']}"
        hash_str += f"@{data['interval']}"
        hash_str += f"#{data['type']}"
        for coord in sorted(data['coordinates'], key=lambda x: (x[0], x[1])):
            hash_str += f"({coord[0]:.4f},{coord[1]:.4f})"

        combined = f"{hashlib.md5(hash_str.encode()).hexdigest()}"
        return combined
    # 检查请求的内容类型是否为JSON
    try:
            # 从数据中提取信息
            startTime=data.get("startTime")
            data_type=data.get("type")
            algorithm=data.get("algorithm")
            interval=data.get("interval")
            coor=data.get("coordinates")

            # 格式化时间
            startTime=datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M")
            if not satisfyTime(startTime,interval):
                raise Exception(f"时间{startTime}应该为{interval}min的整数")
            # 变为yyyyMMddHHmm的时间
            startTime=startTime.strftime("%Y%m%d%H%M")
            # 尝试获取缓存
            redis_key=config.GLOBAL_CONFIG['TRAFFIC_KEY']+startTime+":"+generate_traffic_png_hash(data)
            redisWoker=config.GLOBAL_CONFIG["REDIS_CLIENT"]
            value=redisWoker.getJSON(redis_key)
            if value is not None:
                logger.info(f"获取traffic缓存成功{startTime}")
                return value
            # 缓存未命中,请求数据，但使用分布式锁防止高并发
            token=generateUUID()+"Z"+str(random.randint(0,100000))
            lock=getLock(redis_key,token)
            try:
                if lock:
                    logger.info(f"获取traffic缓存失败，获取锁成功{startTime}:{redis_key}:{token}")
                    # 双重检查机制
                    value=redisWoker.getJSON(redis_key)
                    if value is not None:
                        logger.info(f"获取traffic缓存失败，双重检查成功{startTime}:{redis_key}:{token}")
                        return value
                    
                    # 获得一个红绿灯的数组
                    res=await trafficPoint(coor,startTime,algorithm,data_type,interval=interval)
                    returnData=[]
                    for i in range(len(res)):
                        # 记录一个经纬度点的数据
                        temp={}
                        temp["id"]=i
                        temp["lng"]=coor[i][0]
                        temp["lat"]=coor[i][1]
                        temp["data"]=[]
                        for j in range(len(res[0])):
                            # 记录每个时间帧的数据，一共40帧
                            temp["data"].append(res[i][j])
                        returnData.append(temp)
                    # 为了防止数据太旧了，直接缓存5min
                    redisWoker.setJSON(redis_key,returnData,ex=random_expire(3,10)*60)
                    return returnData
                else:
                    logger.info(f"获取traffic缓存失败，获取锁失败{startTime}:{redis_key}:{token}")
                    # 随机等待10ms-100ms
                    await asyncio.sleep(random.uniform(0.01,0.1))
                    return await trafficPointFromPNG(data)
            finally:
                if lock:
                    logger.info(f"释放traffic锁{startTime}:{redis_key}:{token}")
                    releaseLock(redis_key,token)
    except Exception as e:
        logger.error(f"出现错误:{e}")
        raise e


async def regionCoverage(data):
    '''
    查询区域的覆盖率,一次只会查询一个区域
    请求参数:
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "startTime": "2024-11-04 00:00",
    //时间间隔
    "interval": 6,
    //数据类型
    "type": 0
   //算法编号类型
    "algorithm": 算法名字,
    //覆盖判断阈值
    "threshold": [25,35],
    "id":0,
    //边缘点经度数组
    "coordinates":[[110,20],[113,30],...]
}

    返回值:
    {
    "code": 200,
    "message": "success",
    "data": [
           {
               "id":"string",
               "threshold":25,
               "data":[1,2,3,4,5,...]
           },
           {
               "id":"string",
               "threshold":35,
               "data":[1,2,3,4,5....]
           }
     ]
 }
    '''
    def generate_region_coverage_hash(data):
        # 构造哈希字符串
        hash_str = f"{data['algorithm']}_{data['startTime']}"
        hash_str += f"|{data['id']}"
        hash_str += f"@{data['interval']}"
        hash_str += f"#{'_'.join(map(str, sorted(data['thresholds'])))}"
        for coord in sorted(data['coordinates'], key=lambda x: (x[0], x[1])):
            hash_str += f"({coord[0]:.4f},{coord[1]:.4f})"

        combined = f"{hashlib.md5(hash_str.encode()).hexdigest()}"
        return combined
    # 检查请求的内容类型是否为JSON
    try:
        # 从数据中提取信息
        startTime=data.get("startTime")
        data_type=data.get("type")
        algorithm=data.get("algorithm")
        interval=data.get("interval")
        coor=data.get("coordinates")
        thresholds=data.get("thresholds")
        id=data.get("id")
        # 格式化时间
        startTime=datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M")
        if not satisfyTime(startTime,interval):
            raise Exception(f"时间{startTime}应该为{interval}min的整数")
        # 变为yyyyMMddHHmm的时间
        startTime=startTime.strftime("%Y%m%d%H%M")
        # 尝试获取缓存
        redis_key=config.GLOBAL_CONFIG['COVERAGE_KEY']+startTime+":"+generate_region_coverage_hash(data)
        redisWoker=config.GLOBAL_CONFIG["REDIS_CLIENT"]
        value=redisWoker.getJSON(redis_key)
        if value is not None:
            logger.info(f"获取coverage缓存成功{startTime}")
            return value
        # 缓存未命中,请求数据，但使用分布式锁防止高并发
        token=generateUUID()+"Z"+str(random.randint(0,100000))
        lock=getLock(redis_key,token)
        try:
            if lock:
                logger.info(f"获取coverage缓存失败，获取锁成功{startTime}:{redis_key}:{token}")
                # 双重检查机制
                value=redisWoker.getJSON(redis_key)
                if value is not None:
                    logger.info(f"获取coverage缓存失败，双重检查成功{startTime}:{redis_key}:{token}")
                    return value
                # 获得逐个计算覆盖率
                returnData=[]
                for threshold in thresholds:
                    res=await coverage_rate(coor,startTime,algorithm,threshold,data_type,interval=interval)
                    # 记录一个经纬度点的数据
                    temp={}
                    temp["id"]=id
                    temp["threshold"]=threshold
                    temp["data"]=[]
                    for i in range(len(res)):
                        # 记录每个时间帧的数据，一共40帧
                        temp["data"].append(res[i])
                    returnData.append(temp)
                # 更新缓存
                redisWoker.setJSON(redis_key,returnData,ex=random_expire(3,10)*60)
                # 返回JSON响应
                return returnData
            else:
                logger.info(f"获取coverage缓存失败，获取锁失败{startTime}:{redis_key}:{token}")
                # 随机等待10ms-100ms
                await asyncio.sleep(random.uniform(0.01,0.1))
                return await regionCoverage(data)
        finally:
            if lock:
                logger.info(f"释放coverage锁{startTime}:{redis_key}:{token}")
                releaseLock(redis_key,token)
    except Exception as e:
        logging.error(f"出现错误:{e}")
        raise e

def convectiveTracking(data):
    '''
    请求参数
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "startTime": "2024-11-04 00:00",
    "interval": 6,
    "algorithm": "",
    "poolScale":池化尺度
    }
    返回值
    {
    'algorithm':
    'entities':[
    {
                    "id": global_id,
                    "time": "",
                    "speed": 0,  # 初始化平均速度
                    "direction": None,  # 初始化平均方向
                    "spanData": [{
                        "time": "时间",
                        "outline": [边缘点],
                        "lat": 纬度,
                        "lon": 经度,
                        "y": ,
                        "u": 东西速度,
                        "v": 南北速度,
                        "direction": 方向
                    }]
                }
                ...
    ]
}
    '''
    try:
        # 从数据中提取信息
        startTime=data.get("startTime")
        algorithm=data.get("algorithm")
        interval=data.get("interval")
        poolScale=data.get("poolScale")
        # 格式化时间
        startTime=datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M")
        if not satisfyTime(startTime,interval):
            raise Exception(f"时间{startTime}应该为{interval}min的整数")
        if not satisfyPoolScale(poolScale):
            raise Exception(f"当前的poolScale为{poolScale},不满足要求。poolScale只能为0，4，16")        
        # 变为yyyyMMddHHmm的时间
        startTime=startTime.strftime("%Y%m%d%H%M")
        # 先尝试获取缓存
        # redisWorker=RedisWorker()
        redisWorker = config.GLOBAL_CONFIG["REDIS_CLIENT"]
        key=config.GLOBAL_CONFIG["TRACKING_KEY"]+algorithm+":"+startTime
        value=redisWorker.getJSON(key)
        import time
        st=time.time()
        if value is None:
            logger.info(f"获取tracking缓存失败{startTime}")
            token=generateUUID()+"Z"+str(random.randint(0,100000))
            lock=getLock(key,token)
            try:
                if lock:
                    logger.info(f"获取tracking缓存失败，获取锁成功{startTime}:{key}:{token}")
                    # 双重检查机制
                    value=redisWorker.getJSON(key)
                    if value is not None:
                        logger.info(f"获取tracking缓存失败，双重检查成功{startTime}:{key}:{token}")
                        return value
                    # 获得一个对流云追踪结果
                    res=monomer_tracking(date=startTime,
                                         algorithm=algorithm,
                                         interval_minutes=interval,
                                         poolingScale=poolScale)
                    # 更新缓存,过期时间为30min
                    redisWorker.setJSON(key,res,ex=random_expire(30,60)*60)
                    return res
                else:
                    logger.info(f"获取tracking缓存失败，获取锁失败{startTime}:{key}:{token}")
                    # 随机等待10ms-100ms
                    time.sleep(random.uniform(0.01,0.1))
                    return convectiveTracking(data)
            finally:
                if lock:
                    logger.info(f"释放tracking锁{startTime}:{key}:{token}")
                    releaseLock(key,token)
        else:
            logger.info(f"获取tracking缓存成功{startTime}")
            res=value
        ed=time.time()
        logger.info(f"对流追踪{(ed-st)*1000}ms")
        # 返回JSON响应
        return res
    except Exception as e:
        logger.error(f"error:tracking出现错误:{e}")
        raise e
if __name__=='__main__':
    date=datetime.datetime(year=2024,month=12,day=21,hour=6,minute=0)
    print(satisfyTime(date,60))