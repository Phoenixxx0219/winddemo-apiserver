# 系统后端服务器

## API使用说明
1. 获取真实图片数据
```
/realimage/{date}/{type}/real/{filename}
```
2. 获取预测图片数据
```
/forcastimage/{date}/{type}/forcast/{time}/{filename}
```
3. 获取单体轮廓数据
```
/api/convective/tracking
请求参数：
{
  "time": "2024-06-04 00:00",
  "interval": "6",
  "algorithm": "forcast"
}
```

## 源代码使用说明
查询ip地址：`ipconfig`，使用该ip地址和端口8080即可访问后端服务器
启动后端服务器：`python apiServer.py`

## 日志
- [X] 20250402实现简易版fastapi
- [X] 20250405实现图片数据获取接口
- [X] 20250406实现单体轮廓数据获取接口
- [X] 20250410使用自己训练的模型进行雷达图片的预测，得到预测数据并存入后端服务器
- [X] 20250410接入自己预测的雷达数据，更新图片数据获取接口
- [X] 20250410使用自己预测的雷达数据进行单体轮廓识别，更新单体轮廓数据获取接口
- [X] 20250410实现单点查询预警接口
- [X] 20250411实现lru_cache缓存
- [X] 20250411目前的预测图片与原始图片的色卡有点对不上，重新生成预测图片