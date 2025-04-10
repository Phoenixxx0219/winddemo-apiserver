import json
from datetime import datetime
from function import point_in_polygon, find_earliest_entity

# def point_in_polygon(point, polygon):
#     """
#     利用射线法判断点是否在多边形内
#     参数:
#       point: (lat, lon) 格式的坐标
#       polygon: [(lat1, lon1), (lat2, lon2), ...] 多边形顶点列表
#     返回:
#       True 如果点在多边形内，否则 False
#     """
#     x, y = point
#     inside = False
#     n = len(polygon)
    
#     # 遍历所有边
#     for i in range(n):
#         j = (i + 1) % n  # 保证最后一个点和第一个点构成边界
#         xi, yi = polygon[i]
#         xj, yj = polygon[j]
#         # 判定射线与边相交的条件
#         intersect = ((yi > y) != (yj > y)) and \
#                     (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi)
#         if intersect:
#             inside = not inside
#     return inside

# def find_earliest_entity(json_data, target_point):
#     """
#     根据前端传入的坐标点，查找包含该点的最早出现的单体轮廓
#     参数:
#       json_data: 包含单体轮廓数据的字典对象
#       target_point: (lat, lon) 格式的前端传入坐标
#     返回:
#       (entity_id, occurrence_time) 或者 None
#     """
#     earliest_entity = None
#     earliest_time = None  # 存储 datetime 对象，便于比较
    
#     # 遍历所有单体轮廓
#     for entity in json_data.get("entities", []):
#         entity_id = entity.get("id")
#         # 遍历该单体的每一个检测时刻
#         for span in entity.get("spanData", []):
#             # 获取轮廓 outline 信息，注意 JSON 中每个坐标点的格式是 [lat, lon]
#             polygon = span.get("outline", [])
#             if not polygon or len(polygon) < 3:
#                 continue  # 简单排除非有效多边形
            
#             # 判断前端传入点是否在该轮廓内部
#             if point_in_polygon(target_point, polygon):
#                 # 解析该时刻对应的时间，假定时间格式为 "%Y-%m-%d %H:%M:%S"
#                 try:
#                     span_time = datetime.strptime(span["time"], "%Y-%m-%d %H:%M:%S")
#                 except Exception as e:
#                     continue  # 格式异常则跳过

#                 # 如果该轮廓首次出现的时间更早，则更新 earliest_entity
#                 if earliest_time is None or span_time < earliest_time:
#                     earliest_time = span_time
#                     earliest_entity = {"id": entity_id, "time": span["time"]}
#                 # 该单体中可能存在多个满足条件的时间（但只取最早的即可）
#                 # 可以选择跳出循环，避免多余比较： break
#                 # break    # 如果同一个单体中仅需最早的一个时间，则可使用 break
                
#     return earliest_entity

# 示例数据加载及调用示例
if __name__ == "__main__":
    # 假设 json_string 为前端提供的 JSON 数据
    with open("./demoout/out2.json", 'r') as file:
        data = json.load(file)
    '''
    {
        "algorithm": "radar_difftrans_deploy_3h",
        "entities": [
            {
                "id": 1,
                "time": "2024-06-04 00:00:00",
                "startTime": "2024-06-03 23:06:00",
                "endTime": "2024-06-04 03:00:00",
                "startIndex": 1,
                "endIndex": 40,
                "speed": 34.3396889728348,
                "direction": 45.43736386752073,
                "spanData": [
                    {
                        "time": "2024-06-03 23:06:00",
                        "index": 1,
                        "maxValue": 56.0,
                        "avgValue": 30.474,
                        "outline": [
                            [22.649, 111.912],
                            [22.638, 111.901],
                            [22.638, 111.868],
                            [22.652, 111.870]
                        ],
                        "lat": 21.165,
                        "lon": 112.208,
                        "x": 337.333,
                        "y": 480.254,
                        "u": null,
                        "v": null,
                        "direction": null
                    }
                ]
            }
        ]
    }
    '''
    
    # 前端输入的坐标点 (示例：latitude, longitude)
    target = (23.568692763080577, 114.84766293307796)
    
    result = find_earliest_entity(data, target)
    if result is None:
        print("null")
    else:
        print("找到单体轮廓, id: {}, 出现时间: {}".format(result["id"], result["time"]))
