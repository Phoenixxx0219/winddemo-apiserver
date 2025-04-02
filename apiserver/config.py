# 全局配置类，实现跨模块的数据迁移
GLOBAL_CONFIG={}
def set_config(key,value):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG[key]=value