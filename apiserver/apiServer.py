from dotenv import load_dotenv
import os
import sys
from fastapi import FastAPI
from entity.trafficPointEntity import SingleAlgorithmTrafficPoint,MultiAlgorithmTrafficPoint
from entity.regionEntity import RegionCoverageEntity
from entity.convectiveEntity import ConvectiveTrackingEntity
from entity.lastTimeEntity import LastTimeEntity
from entity.RadarForecastEntity import *
from function import trafficPointFromPNG,regionCoverage,convectiveTracking
from forecasting.pointForecast  import pointForecasting
from forecasting.regionForecast import regionForecasting
from schedule.scheduler import tracking_scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
sys.path.append(os.getcwd())
import config
import argparse
import json
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import asyncio
import uvloop
import asyncio
from cache.cache import RedisWorker
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from logger_config import setup_logging
import redis
logger= setup_logging()
load_dotenv()
# 用户参数
parser=argparse.ArgumentParser()
parser.add_argument("--app_host",default="0.0.0.0",type=str,help="web服务器监听的ip，公网访问需要为0.0.0.0")
parser.add_argument("--app_port",default=9000,type=int,help="web服务器监听的端口")
parser.add_argument("--redis_host",default="localhost",help="Redis服务器的IP，默认为localhost",type=str)
parser.add_argument("--redis_port",default=6379,help="Redis服务器的端口，默认为6379",type=str)
parser.add_argument("--redis_pw",default="ices123456",help="Redis服务器的密码，默认为ices123456",type=str)
parser.add_argument("--pre_compute",default=False,help="获取预计算的结果,将对流追踪提前存储为json",type=bool)
args=parser.parse_args()
# 环境变量优先指定
APP_HOST=os.getenv("APP_HOST",args.app_host)
APP_PORT=os.getenv("APP_PORT",args.app_port)
REDIS_HOST=os.getenv("REDIS_HOST",args.redis_host)
REDIS_PORT=os.getenv("REDIS_PORT",args.redis_port)
REDIS_PW=os.getenv("REDIS_PW",args.redis_pw)
# 统一返回对象 
class Result:
    def __init__(self):
        pass
    def ok(self,data):
        return {
            "code":200,
            "message":"success",
            "data":data
        }
    def error(self,message):
        return {
            "code":500,
            "message":f"{message}",
            "data":""
        }


# 配置fastapi服务器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 初始化key
    config.set_config("TRACKING_KEY","tracking:")
    config.set_config("TRAFFIC_KEY","traffic:")
    config.set_config("RADAR_POINT_KEY",'radar_point_forecasting:')
    config.set_config("RADAR_REGFION_KEY",'radar_region_forecasting:')
    config.set_config("COVERAGE_KEY",'coverage:')
    # 初始化config
    config.set_config("REDISHOST",REDIS_HOST)
    config.set_config("REDISPASS",REDIS_PW)
    config.set_config("REDISPORT",int(REDIS_PORT))
    # 初始化redis连接
    pool=redis.ConnectionPool(host=REDIS_HOST, port=int(REDIS_PORT), password=REDIS_PW, db=0,max_connections=1000)
    redis_client = RedisWorker(pool)
    config.set_config("REDIS_CLIENT", redis_client)
    # 初始化计算的线程池或者进程池
    executors=ThreadPoolExecutor(max_workers=os.cpu_count()*4)
    config.set_config("CPU_POOLS",executors)
    yield
    # await redis_client.close()
    # await redis_client.connection_pool.disconnect()
app = FastAPI(lifespan=lifespan)
# 解决跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#####################################
#####################################
##
##          Fast的API请求
##
#####################################
#####################################
@app.get("/api/ping")
async def read_root():
    logger.info("开始请求ping......")
    return {"Hello": "World"}


@app.post("/api/getLastTime")
async def read_root(item:LastTimeEntity):
    algorithm=item.algorithm
    with open("/data/latest_date.json","r") as f:
        data=json.load(f)
        try:
            time=data[algorithm]
        except Exception as e:
            time=data["real"]

    res=Result().ok({
        "lastTime":time
    })
    return res

@app.post("/api/traffic/singleAlgorithmPoints")
async def getSingleAlgorithmTrafficPoints(item:SingleAlgorithmTrafficPoint):
    '''
    单一算法的多点查询
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "time": "2024-11-04 00:00",
    //时间间隔（根据数据类型确定，后端会对非整interval的startTime向下取整）
    "interval": 6,
    //数据类型
    "dataType": 0
   //算法编号类型
    "algorithm": "算法名字",
    //待查询点
     "points": [
        {"lat": 39.915, "lng": 116.404, id: "a"}, 
        ...
    ],
    }
    '''
    data={

    }
    data["startTime"]=item.time
    data["type"]=item.dataType
    data["algorithm"]=item.algorithm
    data["interval"]=item.interval
    data["coordinates"]=[]
    ids=[]
    # 格式化坐标，将其变为二维数组
    for point in item.points:
         coor=[point['lng'],point['lat']]
         data["coordinates"].append(coor)
         ids.append(point["id"])

    # 红绿灯查询
    try:
        # loop=asyncio.get_event_loop()
        # trafficData=await loop.run_in_executor(executors,trafficPointFromPNG,data)
        trafficData=await trafficPointFromPNG(data)
        responseData={

        }
        for i in range(len(ids)):
            responseData[ids[i]]=trafficData[i]["data"]
        return Result().ok(responseData)
    except Exception as e:
        return Result().error(f"单算法多点红绿灯查询失败,error:{e}")

@app.post("/api/traffic/multiAlgorithmPoint")
async def getMultiAlgorithmTrafficPoints(item:MultiAlgorithmTrafficPoint):
    '''
    多个算法的单点查询
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "time": "2024-11-04 00:00",
    //时间间隔（根据数据类型确定，后端会对非整interval的startTime向下取整）
    "interval": 6,
    //数据类型
    "dataType": 0
   //算法编号类型
    "algorithms": ["算法名字"],
    //待查询点
     "point": {"lat": 39.915, "lng": 116.404, id: "a"}, 

    }
    '''
    responseData={

    }
    for algorithm in item.algorithms:
        data={

        }
        data["startTime"]=item.time
        data["type"]=item.dataType
        data["algorithm"]=algorithm
        data["interval"]=item.interval
        data["coordinates"]=[]
        # 格式化坐标，将其变为二维数组
        point=item.point
        coor=[point['lng'],point['lat']]
        data["coordinates"].append(coor)

        # 红绿灯查询
        try:
            # loop=asyncio.get_event_loop()
            # trafficData=await loop.run_in_executor(executors,trafficPointFromPNG,data)
            trafficData=await trafficPointFromPNG(data)
            responseData[algorithm]=trafficData[0]["data"]
        except Exception as e:
            return Result().error(f"多算法单点红绿灯查询失败,error:{e}")
    return Result().ok(responseData)

@app.post("/api/region/coverage")
async def getRegionCoverage(item:RegionCoverageEntity):
    '''
    区域覆盖率计算
    {   
    //预报时间，后端会返回前1h的真实值，和后3h的单点查询信息
    "time": "2024-11-04 00:00",
    //时间间隔
    "interval": 6,
    "type": 0,
    "algorithm": "算法",
    //覆盖判断阈值
    "thresholds":[25,35],
    //需要计算的区域
    "regions":[
        {   
            //区域的id
            "id":"区域1",
           //边缘点经度数组
            "points":[{"lat": 39.904211, "lng": 116.407395}, {"lat": 39.904211, "lng": 116.407395}, ...]
        },
        ......
    ]
    }
    '''
    responseData={

    }
    responseData["values"]={

    }
    # 异步并行处理region计算
    async def process_region(region):
        data = {
            "startTime": item.time,
            "type": item.dataType,
            "algorithm": item.algorithm,
            "interval": item.interval,
            "thresholds": item.thresholds,
            "id": region["id"],
            "coordinates": [[p['lng'], p['lat']] for p in region["points"]]
        }
        
        try:
            coverage_data = await regionCoverage(data)
            region_resp = {int(obj["threshold"]): obj["data"] for obj in coverage_data}
            return (region["id"], region_resp)
        except Exception as e:
            logger.error(f"区域{region['id']}处理失败: {str(e)}")
            return (region["id"], None)
    
    # 创建并行任务
    tasks = [process_region(region) for region in item.regions]
    results = await asyncio.gather(*tasks)
    responseData["time"]=item.time
    for region_id, region_data in results:
        if region_data is not None:
            responseData["values"][region_id] = region_data
    return Result().ok(responseData)
    
@app.post("/api/convective/tracking")
async def getConvectiveTracking(item:ConvectiveTrackingEntity):
        logger.info(f"开始请求tracking,请求参数:{item}")
        data={

        }
        data["startTime"]=item.time
        data["algorithm"]=item.algorithm
        data["interval"]=item.interval
        data["poolScale"]=item.poolScale
        try:
            loop=asyncio.get_event_loop()
            responseData=await loop.run_in_executor(config.GLOBAL_CONFIG["CPU_POOLS"],convectiveTracking,data)
            # responseData=await convectiveTracking(data)
            if responseData is None:
                Result().error(f"对流追踪请求失败,error:缺少数据{item.time}")
            return Result().ok(responseData)
        except Exception as e:
            return Result().error(f"对流追踪请求失败,error:{e}")

@app.post("/api/forecast/pointForecasting")
async def getRadarPointForecast(item:RadarPointForecastEntity):
    '''
    获取单点的雷达预报结果
    {
    code: "radar_difftrans_deploy_3h",
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
    '''
    logger.info(f"开始处理pointForecasting请求，请求参数:{item}")
    data={

    }
    data["time"]=item.time
    data["code"]=item.code
    data["points"]=item.points
    try:
        # loop=asyncio.get_event_loop()
        # responseData=await loop.run_in_executor(executors,pointForecasting,data)
        responseData=await pointForecasting(data)
        if responseData is None:
            Result().error(f"点雷达预警失败,error:缺少数据{item.time}")     
        return Result().ok(responseData)
    except Exception as e:
        return Result().error(f"点雷达预警失败,error:{e}")

@app.post("/api/forecast/regionForecasting")
async def getRadarRegionForecast(item:RadarRegionForecastEntity):
    '''
    获取区域的雷达预报结果
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
    '''
    logger.info(f"开始处理regionForecasting请求,请求参数{item}")
    data={

    }
    data["time"]=item.time
    data["code"]=item.code
    data["areas"]=item.areas
    try:
        # loop=asyncio.get_event_loop()
        # responseData=await loop.run_in_executor(executors,regionForecasting,data)
        responseData=await regionForecasting(data)
        if responseData is None:
            Result().error(f"点雷达预警失败,error:缺少数据{item.time}")     
        return Result().ok(responseData)
    except Exception as e:
        return Result().error(f"点雷达预警失败,error:{e}")
if __name__=='__main__':
    logger.info("==============================apiServer启动==============================")
    # 初始化key
    config.set_config("TRACKING_KEY","tracking:")
    config.set_config("TRAFFIC_KEY","traffic:")
    config.set_config("RADAR_POINT_KEY",'radar_point_forecasting:')
    config.set_config("RADAR_REGFION_KEY",'radar_region_forecasting:')
    config.set_config("COVERAGE_KEY",'coverage:')
    # 初始化config
    config.set_config("REDISHOST",REDIS_HOST)
    config.set_config("REDISPASS",REDIS_PW)
    config.set_config("REDISPORT",int(REDIS_PORT))
    # 启动定时器
    scheduler = BackgroundScheduler()
    scheduler.add_job(tracking_scheduler, 'interval', minutes=6)  # 每 6 分钟执行一次
    scheduler.start()
    # 启动服务器
    uvicorn.run("apiServer:app", 
                host=APP_HOST, 
                port=int(APP_PORT),
                workers=os.cpu_count(),
                loop="uvloop",
                http="httptools",
                limit_max_requests=10000,
                timeout_keep_alive=60)