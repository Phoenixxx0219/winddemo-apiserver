import cv2
import numpy as np
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
from aiofile import async_open
import config
def convert(array):
#经纬度转数组坐标,不作限制
    left = 108.505
    down = 19.0419
    dx = 0.0109756097560976
    dy = 0.010144927536231883
    idx_array = np.zeros((len(array),2),dtype = int)
    for i,(lon,lat) in enumerate(array):
        lon_j = round((lon - left) / dx)
        lat_i = 690 - round((lat - down) / dy)
        idx_array[i] = [lon_j,lat_i] #和单点查询相反因为fillpoly是先列后行
    return idx_array

async def coverage_rate(array, date, algorithm, threshold, datatype=12, path="/data/Traffic/image", interval=6):
    
    array = convert(array)
    x, y, w, h = cv2.boundingRect(array)
    array = array - [x, y]
    if algorithm == "radar_real_3h":
        algorithm = "real"
    # 构建路径
    pred_path = Path(path) / f"{date[:-4]}/{datatype}/{algorithm}/{date[-4:-2]}-{date[-2:]}"
    if algorithm == 'real':
        pred_path = pred_path.parent  # 去掉最后一级目录

    # 初始化结果数组
    result = np.full(40, -1, dtype=np.float32)

    # 定义处理单个时间点的函数
    async def process_time_point(current_date, result_idx):
        date_str = current_date.strftime("%Y%m%d%H%M")
        img_path = pred_path / f"{date_str}.png" if result_idx >= 10 else Path(path) / f"{date_str[:-4]}/{datatype}/real/{date_str}.png"
        if img_path.exists():
            async with async_open(str(img_path), 'rb') as afp:
                content = await afp.read()
                def decode_and_process():
                    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is None: return -1
                    try:
                        sub_img = img[y:y+h, x:x+w]
                        sub_img = np.where(sub_img > threshold, sub_img, 0)
                        mask = np.zeros(sub_img.shape, dtype=np.uint8)
                        cv2.fillConvexPoly(mask, array, 255)
                        area_mask = np.count_nonzero(mask)
                        if area_mask == 0: return -1
                        area_rada = np.count_nonzero(cv2.bitwise_and(sub_img, sub_img, mask=mask))
                        return area_rada / area_mask
                    except IndexError:
                        return -1
                rate = await asyncio.get_event_loop().run_in_executor(config.GLOBAL_CONFIG['CPU_POOLS'], decode_and_process)
                if rate is not None:
                    result[result_idx] = rate
    
    # 时间点生成优化
    base_date = datetime.datetime.strptime(date, "%Y%m%d%H%M") - datetime.timedelta(hours=1)
    time_points = [
        (base_date + datetime.timedelta(minutes=interval*(i+1)), i)
        for i in range(40)
    ]

    # 批量创建任务
    tasks = [process_time_point(tp[0], tp[1]) for tp in time_points]
    await asyncio.gather(*tasks)

    return result.tolist()


if __name__ == '__main__':
    async def main():
        array = [[113.47906708162985,22.87870360512357],
                 [113.32802176834265,22.62671774799317],
                 [113.55047032063834,21.990281212081406],
                 [113.70014249471383,22.155702604125484],
                 [113.66306773599788,22.230714452223136],
                 [113.56145543433192,22.535432768079048],
                 [113.47906708162985,22.87870360512357]]
        import time
        #记录开始时间
        start_time = time.perf_counter()

        date = "202408160800"
        algorithm = 'radar_real_3h'
        threshold = 25.0
        data=await coverage_rate(array, date, algorithm, threshold,)
        print(data)
        # 记录结束时间
        end_time = time.perf_counter()

        # 计算并打印执行时间
        execution_time = end_time - start_time
        print(f"The test function took {execution_time} seconds to complete.")
    asyncio.run(main())
