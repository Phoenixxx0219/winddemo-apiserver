import datetime
import pytz
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
        import time
        st=time.time()
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
    
def point_in_polygon(point, polygon):
    """
    利用射线法判断点是否在多边形内
    参数:
      point: (lat, lon) 格式的坐标
      polygon: [(lat1, lon1), (lat2, lon2), ...] 多边形顶点列表
    返回:
      True 如果点在多边形内，否则 False
    """
    x, y = point
    inside = False
    n = len(polygon)
    
    # 遍历所有边
    for i in range(n):
        j = (i + 1) % n  # 保证最后一个点和第一个点构成边界
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # 判定射线与边相交的条件
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi)
        if intersect:
            inside = not inside
    return inside

def find_earliest_entity(json_data, target_point):
    """
    根据前端传入的坐标点，查找包含该点的最早出现的单体轮廓
    参数:
      json_data: 包含单体轮廓数据的字典对象
      target_point: (lat, lon) 格式的前端传入坐标
    返回:
      {"id": entity_id, "time": occurrence_time_in_beijing} 或者 None
    """
    earliest_entity = None
    earliest_time = None  # 存储 datetime 对象，便于比较

    # 定义 UTC 和 北京时区（Asia/Shanghai）
    utc_tz = pytz.utc
    beijing_tz = pytz.timezone("Asia/Shanghai")
    
    # 遍历所有单体轮廓
    for entity in json_data.get("entities", []):
        entity_id = entity.get("id")
        # 遍历该单体的每一个检测时刻
        for span in entity.get("spanData", []):
            # 获取轮廓 outline 信息，注意 JSON 中每个坐标点的格式是 [lat, lon]
            polygon = span.get("outline", [])
            if not polygon or len(polygon) < 3:
                continue  # 简单排除非有效多边形

            # 判断前端传入点是否在该轮廓内部
            if point_in_polygon(target_point, polygon):
                # 解析该时刻对应的时间，假定时间格式为 "%Y-%m-%d %H:%M:%S"，且原始时间为 UTC 时间
                try:
                    span_time = datetime.datetime.strptime(span["time"], "%Y-%m-%d %H:%M:%S")
                    # 设置原始时间为 UTC
                    span_time = utc_tz.localize(span_time)
                except Exception as e:
                    continue  # 时间格式异常则跳过

                # 如果该轮廓首次出现的时间更早，则更新 earliest_entity
                if earliest_time is None or span_time < earliest_time:
                    earliest_time = span_time
                    earliest_entity = {"id": entity_id, "time": span["time"]}
                # 同一个单体中只取最早的一个时间，直接跳出循环
                break

    if earliest_entity and earliest_time:
        # 将UTC时间转换为北京时间（Asia/Shanghai）
        beijing_time = earliest_time.astimezone(beijing_tz)
        # 返回格式化后的北京时间字符串，格式保持 "%Y-%m-%d %H:%M:%S"
        earliest_entity["time"] = beijing_time.strftime("%Y-%m-%d %H:%M:%S")

    return earliest_entity