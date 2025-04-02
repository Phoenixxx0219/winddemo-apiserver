# 气象雷达图像序列中的显著性区块跟踪
对气象雷达图像中的显著性区块进行识别、跟踪，得到单体轮廓、移动方向和速度等信息。


## 使用说明
**`recognition.py`**
从气象雷达图像中提取显著性区块的边缘图像edges和雷达反射率reflectivity。
- `edge_recognition_0`：基于形态学操作的显著性区块分割，阈值为1（分辨率为820×690）
- `edge_recognition_4`：基于形态学操作的显著性区块分割，阈值为1（分辨率为206×173）
- `edge_recognition_16`：基于形态学操作的显著性区块分割，阈值为1（分辨率为52×44）

**`tracking.py`**
运用椭圆描述的面积重叠法对单体进行追踪和预测，得到连续两帧之间显著性区块的对应关系relationships。
- `get_ellipses_and_contours`：从边缘图像中提取轮廓并拟合椭圆，返回椭圆参数ellipses与轮廓坐标contours_list，同时计算每个轮廓对应区域内雷达反射率的最大值max_values和平均值avg_values。
- `calculate_contour_area`：计算闭合轮廓的面积。
- `calculate_contour_area_overlap`：计算两个闭合轮廓面积（s1，s2）的交并比，即(s1∩s2)/min(s1, s2)。
- `determine_ellipse_relationships`：运用面积重叠法判断单体间的关系，若两个单体之间的交并比大于0.4，则判断两个单体匹配。若一个单体匹配了多个单体，则其关系为分裂或合并；若一个单体只与一个单体对应，则关系为延续；若一个单体与0个单体匹配，则关系为生成或消散。

**`predict.py`**
计算预测的速度speed（u, v）与方向angle，其中方向是以正北为0°，顺时针360°。
- `getSpeed`：输入两个椭圆的中心点与时间间隔，计算预测的移动速度。
- `calculate_angle`：计算向量的角度值，其中方向是以正北为0°，顺时针360°。
- `getDirection`：输入两个椭圆的中心点，计算预测的移动方向。

**`transformation.py`**
用于将像素坐标转换为经纬度坐标。
- `create_lookup_table`：根据图片的经纬度范围生成查表数组，用于像素点坐标到经纬度的转换。
- `get_latlon_from_coordinates`：将坐标(x, y)转换为对应的经纬度(lat, lon)。
- `convert_outlines_to_latlon`：将轮廓坐标outlines转换为对应的经纬度轮廓坐标latlon_outlines。

**`convective.py`**
处理输入输出。
- `batch_process`：输入date、algorithm和poolingScale，查询对应时间的10张真实图片与30张预测图片（均为已经处理过的灰度图，像素值即为雷达反射率），然后调用`edge_recognition`函数提取轮廓图像edges和雷达反射率reflectivity。
- `add_span_data`：在一个单体entity中添加一帧的信息spanData。
- `add_entity`：添加一个新的单体信息entity。
- `monomer_tracking`：输入date、algorithm和poolingScale（默认原图为0），首先调用`batch_process`函数获取轮廓图像edges、雷达反射率reflectivity以及第一张图片对应的时间start_time，然后每连续的两帧图片为一组，调用`get_ellipses_and_contours`进行椭圆拟合，接着调用`determine_ellipse_relationships`获取两帧图像中单体的变化关系relationships，然后按照输出格式得到最终的output_data。


## 返回值格式
output_data (dict): 
```python
{
    algorithm: '算法名',
    entities: [
        {
            id: 1,    # 单体编号，从1开始
            time: "2024-06-15 07:00:00",    # 请求时间，即第10张图片的时间
            startTime: "2024-06-15 06:06:00",   # 开始时间，即单体第一次出现的时间
            endTime: "2024-06-15 06:12:00", # 结束时间，即单体最后一次出现的时间
            startIndex: 1,  # 开始时间对应的索引值
            endIndex: 2,    # 结束时间对应的索引值
            speed: null,    # 预测的单体移动速度，由开始和结束的坐标值计算得到
            direction: null,    # 预测的单体移动方向，由开始和结束的坐标值计算得到
            spanData: [
                {
                    time: "2024-06-15 06:06:00",    # 当前帧对应的时间
                    index: 1,   # 当前帧对应的时间索引值
                    maxValue: 52.0, # 单体的最大反射率
                    avgValue: 24.734,   # 单体的平均反射率
                    outline: [],    # 单体轮廓坐标
                    lat: 123,   # 单体中心点的经纬度
                    lon: 123,
                    x: 1,       # 单体中心点对应图片上的坐标
                    y: 2,
                    u: 10,      # 单体的速度，由这一帧和下一帧的中心值计算得到，若下一帧没有值，则为null
                    v: 1,
                    direction: 90  # 单体的移动方向，以正北为0，顺时针360
                },
                {
                    time: '2024-06-15 06:12:00',
                    outline: [],
                    lat: 123,
                    lon: 123,
                    x: 1,
                    y: 2,
                    u: 10,
                    v: 1,
                    direction: 'E'
                }
            ],
        }
    ]
}
```