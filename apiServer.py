from fastapi import FastAPI  # FastAPI 是一个为你的 API 提供了所有功能的 Python 类。
import uvicorn

#创建应用程序，app是应用程序名
app = FastAPI()  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用

# API请求

# 异步的请求参数，函数加上async。针对什么路由，就写上什么路径
@app.get("/api/ping")
async def read_root():
    return {"Hello": "World"}

# @app.post("/api/convective/tracking")
# async def getConvectiveTracking(item:ConvectiveTrackingEntity):
#         data={

#         }
#         data["startTime"]=item.time
#         data["algorithm"]=item.algorithm
#         data["interval"]=item.interval
#         data["poolScale"]=item.poolScale
#         try:
#             loop=asyncio.get_event_loop()
#             responseData=await loop.run_in_executor(config.GLOBAL_CONFIG["CPU_POOLS"],convectiveTracking,data)
#             # responseData=await convectiveTracking(data)
#             if responseData is None:
#                 Result().error(f"对流追踪请求失败,error:缺少数据{item.time}")
#             return Result().ok(responseData)
#         except Exception as e:
#             return Result().error(f"对流追踪请求失败,error:{e}")



if __name__ == '__main__':
    #注意，run的第一个参数 必须是文件名:应用程序名
    uvicorn.run("apiServer:app", port=8080, reload=True)
