# 实现后台定时任务
import sys
import os
sys.path.append(os.getcwd())
from apscheduler.schedulers.background import BackgroundScheduler
import time
from datetime import datetime, timedelta,timezone
# 定义你的定时任务函数
from apscheduler.schedulers.background import BackgroundScheduler
import time
from tracking.convective import monomer_tracking
from cache.cache import RedisWorker
import config
import json
import logging
logger=logging.getLogger('apiServer')
def mytask():
    dt=datetime.now()
    print(f"定时任务启动{dt}")
def nearest_minute(dt,interval):
    '''
    当前时刻向下取整获取整interval时刻
    '''
    # 计算距离当前时间的分钟数
    minutes = dt.minute
    # 计算需要添加或减去的时间量，使其成为6的倍数
    remainder = minutes % interval
    if remainder == 0:
        # 如果已经是6的倍数，则不需要调整
        return dt
    else:
        # 否则，直接减去余数
        return dt - timedelta(minutes=remainder)
# 定义你的定时任务函数
def tracking_scheduler():
    '''
    定期预计算tracking的结果
    '''    
    algorithms=[
        'real',
        'radar_difftrans_deploy_3h',
        'radar_cotrec_3h',
        'radar_opticalflow_3h',
        'radar_gru_deploy_3h',
        'radar_trans_deploy_3h'
    ]
    logger.info("===================开始更新缓存===================")
    with open("/data/latest_date.json","r") as f:
        data=json.load(f)
    for algorithm in algorithms:
        # 获取最新时间
        startTime=data[algorithm]
        startTime=datetime.strptime(startTime,'%Y%m%d %H:%M')
        startTime=startTime-timedelta(hours=8)
        if algorithm=='real':
            startTime=startTime-timedelta(hours=3)
        # startTime=datetime(year=2024,month=6,day=15,hour=10,minute=0)
        interval=6
        startTime=nearest_minute(startTime,interval)
        for i in range(0,5):
            # 后台每3min update之前30min结果，防止遗漏
            formatTime=startTime-timedelta(minutes=i*interval)  
            # 变为yyyyMMddHHmm的时间
            inputTime=formatTime.strftime("%Y%m%d%H%M")
            # 先尝试获取缓存
            redisWorker=RedisWorker()
            key=config.GLOBAL_CONFIG.get("TRACKING_KEY")+algorithm+":"+inputTime
            # 获得一个对流云追踪结果
            res=monomer_tracking(date=inputTime,
                                 algorithm=algorithm,
                                 interval_minutes=interval)
            # 更新缓存,过期时间为30min
            redisWorker.setJSON(key,res,ex=30*60)
            logger.info(f"更新tracking缓存:{key}")
    logger.info("===================更新缓存完毕===================")

if __name__=='__main__':
    print("调度器已启动。")
    # 创建后台调度器
    scheduler = BackgroundScheduler()

    # 添加定时任务，例如每3min执行一次
    scheduler.add_job(tracking_scheduler, 'interval', seconds=10)

    # 启动调度器
    scheduler.start()
    # 为了防止脚本退出，可以在这里加入一个阻塞调用
    try:
        # 这里使用无限循环来保持脚本运行
        while True:
            time.sleep(2)
    except (KeyboardInterrupt, SystemExit):
        # 当你按下Ctrl+C或者执行退出操作时，关闭调度器
        scheduler.shutdown()