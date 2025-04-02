# redis缓存默认使用string类型，后续可以优化一下毕竟string的存储太让人崩溃
import redis
import json
import sys
import os
sys.path.append(os.getcwd())
import config
import logging
logger=logging.getLogger('apiServer')
sys.path.append(os.getcwd())
import config
import random
import uuid
def generateUUID():
    return str(uuid.uuid4())
def getLock(key,token):
    '''
    token时uuid，为每一把锁附上唯一标识，防止误删
    '''
    lock_key=f"lock:{key}"
    redisworker=config.GLOBAL_CONFIG['REDIS_CLIENT']
    flag=redisworker.setNX(lock_key,token,ex=random_expire(10,20))
    return flag if flag else False
def releaseLock(key,token):
    '''
    释放锁,需要检查锁是否拥有
    '''
    lock_key=f"lock:{key}"
    redisworker=config.GLOBAL_CONFIG['REDIS_CLIENT']
    # 使用 Lua 脚本保证原子性
    script = """
    if redis.call("get", KEYS[1]) == ARGV[1] then
        return redis.call("del", KEYS[1])
    else
        return 0
    end
    """
    redisworker.eval(script, 1, lock_key, token)
def random_expire(low,high):
    '''
    生成随机数作为expire时间
    '''
    data=random.randint(low,high)
    return data
class RedisWorker:
    def __init__(self,pool=None):
        try:
            if pool is None:
                logger.info("不启动连接池")
                self.r = redis.Redis(host=config.GLOBAL_CONFIG.get("REDISHOST"), port=config.GLOBAL_CONFIG.get("REDISPORT"), db=0, password=config.GLOBAL_CONFIG.get("REDISPASS"))
            else:
                logger.info("启动连接池")
                self.r = redis.Redis(connection_pool=pool)  # 使用全局连接池
        except Exception as e:
            logger.error(f"redis连接失败,{e}")
    def setNX(self,key,value,ex=-1):
        return self.r.set(key,value,nx=True,ex=ex)
    def delKey(self,key):
        self.r.delete(key)
    def eval(self,script,numkeys,*args):
        self.r.eval(script,numkeys,*args)
    def setJSON(self,key,value,ex=-1):
        try:
            if not isinstance(value,str):
                value=json.dumps(value,ensure_ascii=False )
            if ex<=0:
                self.r.set(key,value)
            else:
                self.r.set(key,value,ex=ex)
        except Exception as e:
            logger.error(f"redis宕机:{e}")
            return
    def getJSON(self,key):
        try:
            data=self.r.get(key)
            if data is None:
                return data
            else:
                return json.loads(data.decode("utf-8"))
        except Exception as e:
            logger.error(f"redis宕机:{e}")
            return None
    def delKey(self,key):
        self.r.delete(key)
if __name__=='__main__':
    # 初始化key
    config.set_config("TRACKING_KEY","tracking:")
    # 初始化config
    config.set_config("REDISHOST","localhost")
    config.set_config("REDISPASS","ices123456")
    config.set_config("REDISPORT",6379)
    config.set_config("REDIS_CLIENT",RedisWorker())
    key="testLock:20250307"
    token="123456789"
    releaseLock(key,token)
    flag=getLock(key,token)
    print(flag)
    releaseLock(key,token)
    flag=getLock(key,token)
    print(flag)
    releaseLock(key,"23456789")
    flag=getLock(key,token)
    print(flag)
    # key="tracking:difftrans_deploy_3h:202412172000:entity"
    # data={
    #     "algo":'dadwadwa',
    #     "data":456,
    #     "test":[
    #         [1,2],
    #         [3,4]
    #     ]
    # }

    # redisworker=RedisWorker()
    # redisworker.setJSON(key,data)
    # print(redisworker.getJSON(key))