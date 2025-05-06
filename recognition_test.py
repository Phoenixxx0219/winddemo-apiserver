import cv2
import math
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt

def abstract_color(image_path):
    """
    处理图片，将其HSV颜色映射到对应的反射率。

    参数：
        image_path (str): 输入图片的路径
    返回值：
        np.ndarray: 处理后的图片，像素值为对应的反射率*2
    """
    # 定义 HSV 到反射率的映射关系
    hsv_to_reflection = [([90, 255, 236], 7),
                         ([101, 254, 246], 12),
                         ([120, 255, 246], 17),
                         ([60, 255, 239], 22),
                         ([60, 255, 200], 27),
                         ([60, 255, 144], 32),
                         ([30, 255, 255], 37),
                         ([25, 255, 231], 42),
                         ([17, 253, 255], 47),
                         ([0, 255, 255], 52),
                         ([0, 255, 166], 57),
                         ([0, 255, 101], 62),
                         ([150, 255, 255], 67),
                         ([138, 147, 201], 72)]
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, (820, 690))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total_img = np.zeros((image.shape[0],image.shape[1]))   #占位初始化为0
    for i,rflct_rate in hsv_to_reflection:  #遍历每个颜色及其对应的反射率
        mask = cv2.inRange(hsv, np.array(i), np.array(i))   #按颜色取掩膜
        filtered_img = cv2.bitwise_and(image, image, mask=mask)     #提取图像中的颜色
        gray_img = cv2.cvtColor(filtered_img,cv2.COLOR_BGR2GRAY)    #转灰度图
        gray_img = np.array(gray_img, dtype = np.float32)   #整形转转浮点型
        gray_img[gray_img > 0] = rflct_rate     #将大于1的转为对应颜色的反射率
        total_img += gray_img   #汇集到一起
    
    return total_img

def get_area_threshold(size):
    """
    计算实际面积对应图片中的面积大小
    """
    # 图片总面积
    earth_radius_km=6371
    left, right = 108.505, 117.505
    up, down = 26.0419, 19.0419
    delta_lon = math.radians(abs(right - left))
    up_rad = math.radians(up)
    down_rad = math.radians(down)
    delta_sin_lat = math.sin(up_rad) - math.sin(down_rad)
    area = (earth_radius_km ** 2) * delta_lon * delta_sin_lat
    # 对应阈值面积
    width, height = 820, 690
    threshold = size * width * height / area
    return int(threshold)

def edge_recognition(image):
    """
    基于形态学操作的显著性区块分割与边缘识别，阈值为30
    """
    # 读入图像，像素值为雷达反射率
    reflectivity = image
    # 二值化处理，将低于30dBZ的反射率置为0，其他置为255
    _, binary_image = cv2.threshold(reflectivity, 30, 255, cv2.THRESH_BINARY)
    # 1. 形态学闭操作，连接相邻高回波区域
    kernel = np.ones((7, 7), np.uint8)
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # 2. 去除小的高回波区域，面积阈值为500
    closed_image = np.uint8(closed_image)  # 转换为uint8类型
    # 查找轮廓并筛选掉小区域
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(closed_image)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 只有面积大于500的显著性区块
            cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
    # 3. Canny边缘检测
    blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0) # 先进行高斯模糊减少噪声
    edges = cv2.Canny(blurred_image, 50, 150)
    # 4. 补全四周的像素值
    edges[0, :] = blurred_image[0, :]
    edges[-1, :] = blurred_image[-1, :]
    edges[:, 0] = blurred_image[:, 0]
    edges[:, -1] = blurred_image[:, -1]

    return edges, reflectivity

def get_ellipses_and_contours(binary_image, reflectivity, area_threshold=15):
    """
    从图像中提取轮廓并拟合椭圆，计算每个轮廓对应区域内的最大值和平均值。
    """
    height, width = binary_image.shape[:2]  # 获取图片的高度和宽度，用于后续判断椭圆中心是否超出范围
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    # 找到轮廓
    contours, _ = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ellipses = []
    contours_list = []
    max_values = []
    avg_values = []

    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:
            # 确保轮廓闭合
            contour = np.vstack([contour, [contour[0]]]) if not np.array_equal(contour[0], contour[-1]) else contour
            # 轮廓近似为多边形
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx_polygon) >= 5:
                ellipse = cv2.fitEllipse(approx_polygon)
                (center_x, center_y), (major_axis, minor_axis), angle = ellipse
                # 判断椭圆中心是否超出图片范围
                if 0 <= center_x < width and 0 <= center_y < height:
                    ellipses.append(ellipse)
                    # 转换轮廓为二维数组并闭合
                    contour_2d = contour[:, 0, :]
                    closed_contour = np.vstack([contour_2d, contour_2d[0]])
                    contours_list.append(closed_contour.tolist())
                    # 计算轮廓内的最大值和平均值
                    mask = np.zeros_like(binary_image, dtype=np.uint8)
                    cv2.drawContours(mask, [closed_contour], -1, 255, thickness=cv2.FILLED)
                    # 使用掩膜来提取 reflectivity 中轮廓区域的值
                    masked_reflectivity = cv2.bitwise_and(reflectivity, reflectivity, mask=mask)
                    # 计算最大值和平均值
                    max_value = np.max(masked_reflectivity)
                    avg_value = np.mean(masked_reflectivity[masked_reflectivity > 0])  # 只考虑非零像素
                    max_values.append(max_value)
                    avg_values.append(avg_value)
                else:
                    continue  # 如果超出范围，直接舍弃这个椭圆，不添加到结果列表中

    return ellipses, contours_list, max_values, avg_values

def visualize_ellipses_and_closed_contours(image_path, ellipses, grouped_closed_coordinates):
    # 读取原始图像
    image = cv2.imread(image_path)
    color_image = cv2.resize(image, (820, 690))

    # 绘制拟合的椭圆和闭合轮廓
    for i, (ellipse, closed_coordinates) in enumerate(zip(ellipses, grouped_closed_coordinates)):
        # 绘制椭圆
        cv2.ellipse(color_image, ellipse, (204, 153, 255), 2)
        
        # 绘制闭合轮廓
        closed_coordinates_np = np.array(closed_coordinates, dtype=np.int32)
        cv2.polylines(color_image, [closed_coordinates_np], isClosed=True, color=(0, 0, 255), thickness=2)
        
        # 在椭圆中心绘制编号
        # center = (int(ellipse[0][0]), int(ellipse[0][1]))  # 椭圆中心坐标
        # cv2.putText(color_image, str(i+1), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return color_image


origin_paths = ["./static/202406011300.png", "./static/202406040006.png"]
for origin_path in origin_paths:
    gray_img = abstract_color(origin_path)
    edges, reflectivity = edge_recognition(gray_img)
    ellipses, contours_list, max_values, avg_values = get_ellipses_and_contours(edges, reflectivity)
    color_img = visualize_ellipses_and_closed_contours(origin_path, ellipses, contours_list)
    result_path = origin_path.replace('.png', '_result.png')
    cv2.imwrite(result_path, color_img)