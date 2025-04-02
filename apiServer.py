from fastapi import FastAPI  # FastAPI 是一个为你的 API 提供了所有功能的 Python 类。
import uvicorn

#创建应用程序，app是应用程序名
app = FastAPI()  # 这个实例将是创建你所有 API 的主要交互对象。这个 app 同样在如下命令中被 uvicorn 所引用

#异步的请求参数，函数加上async。针对什么路由，就写上什么路径
@app.get("/")
async def home():
    return {"user_id": 1002}


@app.get("/shop")
async def shop():
    return {"shop": "商品信息"}


if __name__ == '__main__':
    #注意，run的第一个参数 必须是文件名:应用程序名
    uvicorn.run("apiServer:app", port=8080, reload=True)
