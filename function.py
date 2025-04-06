import datetime

from tracking.convective import monomer_tracking

def satisfyTime(dt,interval:int):
    min=dt.minute
    if min%interval==0:
        return True
    else:
        return False

def convectiveTracking(data):
    '''
    请求参数
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "startTime": "2024-11-04 00:00",
    "interval": 6,
    "algorithm": ""
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
        # 格式化时间
        startTime=datetime.datetime.strptime(startTime,"%Y-%m-%d %H:%M")
        if not satisfyTime(startTime,interval):
            raise Exception(f"时间{startTime}应该为{interval}min的整数")      
        # 变为yyyyMMddHHmm的时间
        startTime=startTime.strftime("%Y%m%d%H%M")
        # 先尝试获取缓存
        # redisWorker=RedisWorker()
        # redisWorker = config.GLOBAL_CONFIG["REDIS_CLIENT"]
        # key=config.GLOBAL_CONFIG["TRACKING_KEY"]+algorithm+":"+startTime
        # value=redisWorker.getJSON(key)
        import time
        st=time.time()
        # if value is None:
        #     print(f"获取tracking缓存失败{startTime}")
        #     token=generateUUID()+"Z"+str(random.randint(0,100000))
        #     lock=getLock(key,token)
        #     try:
        #         if lock:
        #             print(f"获取tracking缓存失败，获取锁成功{startTime}:{key}:{token}")
        #             # 双重检查机制
        #             value=redisWorker.getJSON(key)
        #             if value is not None:
        #                 print(f"获取tracking缓存失败，双重检查成功{startTime}:{key}:{token}")
        #                 return value
        #             # 获得一个对流云追踪结果
        #             res=monomer_tracking(date=startTime,
        #                                  algorithm=algorithm,
        #                                  interval_minutes=interval,
        #                                  poolingScale=poolScale)
        #             # 更新缓存,过期时间为30min
        #             redisWorker.setJSON(key,res,ex=random_expire(30,60)*60)
        #             return res
        #         else:
        #             print(f"获取tracking缓存失败，获取锁失败{startTime}:{key}:{token}")
        #             # 随机等待10ms-100ms
        #             time.sleep(random.uniform(0.01,0.1))
        #             return convectiveTracking(data)
        #     finally:
        #         if lock:
        #             print(f"释放tracking锁{startTime}:{key}:{token}")
        #             releaseLock(key,token)
        # else:
        #     print(f"获取tracking缓存成功{startTime}")
        #     res=value
        res=monomer_tracking(date=startTime,
                             algorithm=algorithm,
                             interval_minutes=interval)
        
        ed=time.time()
        print(f"对流追踪{(ed-st)*1000}ms")
        # 返回JSON响应
        return res
    except Exception as e:
        print(f"error:tracking出现错误:{e}")
        raise e