# 红绿灯相关的python代码
import cv2
import numpy as np
import datetime
import os
from typing import List
import sys
import numpy as np
import cv2
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from aiofile import async_open
import asyncio
import config
async def trafficPoint(array, date, algorithm, datatype=12, path="/data/Traffic/image", interval=6):
    def decode_image(content):
        return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
    '''
    单个算法多点查询
    '''
    # 经纬度转数组坐标,不作限制
    left = 108.505
    down = 19.0419
    dx = 0.0109756097560976
    dy = 0.010144927536231883

    # 提前计算坐标转换
    idx_array = np.array([(690 - round((lat - down) / dy), round((lon - left) / dx)) for lon, lat in array], dtype=int)
    if algorithm == "radar_real_3h":
        algorithm = "real"
    # 构建路径
    pred_path = Path(path) / f"{date[:-4]}/{datatype}/{algorithm}/{date[-4:-2]}-{date[-2:]}"
    if algorithm == 'real':
        pred_path = pred_path.parent  # 去掉最后一级目录

    # 初始化结果数组
    result = np.full((len(array), 40), -1, dtype=np.float32)

    # 定义处理单个时间点的函数
    async def process_time_point(current_date, result_col):
        date_str = current_date.strftime("%Y%m%d%H%M")
        img_path = pred_path / f"{date_str}.png" if result_col >= 10 else Path(path) / f"{date_str[:-4]}/{datatype}/real/{date_str}.png"
        
        if img_path.exists():
            # data = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            # 异步读取
            async with async_open(str(img_path), 'rb') as afp:
                content = await afp.read()
                data = await asyncio.get_event_loop().run_in_executor(
                config.GLOBAL_CONFIG['CPU_POOLS'],
                decode_image,
                content
                )
                if data is not None:
                    for idx, (i, j) in enumerate(idx_array):
                        if 0 <= i < data.shape[0] and 0 <= j < data.shape[1]:
                            result[idx][result_col] = data[i][j]

    # 创建异步任务列表
    tasks = []
    current_date = datetime.datetime.strptime(date, "%Y%m%d%H%M") - datetime.timedelta(hours=1)
    
    # 处理前10个时间点（real）
    for real_id in range(10):
        current_date += datetime.timedelta(minutes=interval)
        tasks.append(process_time_point(current_date, real_id))

    # 处理后30个时间点（pred）
    for pred_id in range(10, 40):
        current_date += datetime.timedelta(minutes=interval)
        tasks.append(process_time_point(current_date, pred_id))
    
    # 异步执行所有任务
    await asyncio.gather(*tasks)

    return result.tolist()

if __name__=='__main__':
    import time
    import asyncio
    
    async def main():
        start_time = time.perf_counter()
        array = [[113.47906708162985,22.87870360512357],
                [113.32802176834265,22.62671774799317],
                [113.55047032063834,21.990281212081406],
                [113.70014249471383,22.155702604125484],
                [113.66306773599788,22.230714452223136],
                [113.56145543433192,22.535432768079048],
                [113.47906708162985,22.87870360512357]]
        date = "202406120500"
        algorithm = 'radar_difftrans_deploy_3h'
        data = await trafficPoint(array, date, algorithm)
        print(data)
        print(len(data[0]))
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The test function took {execution_time} seconds to complete.")
    
    # 创建并运行事件循环
    asyncio.run(main())
