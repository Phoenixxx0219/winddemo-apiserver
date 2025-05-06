import cv2
import math
import json
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt


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

def edge_recognition_0(image_path, area_threshold, reflectivity_threshold):
    """
    基于形态学操作的显著性区块分割与边缘识别，阈值为20（分辨率为820×690）
    """
    # 读入图像，像素值为雷达反射率
    reflectivity = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化处理，将低于20dBZ的反射率置为0，其他置为255
    _, binary_image = cv2.threshold(reflectivity, reflectivity_threshold, 255, cv2.THRESH_BINARY)
    # 去除小的高回波区域，面积阈值为1500
    closed_image = np.uint8(binary_image)  # 转换为uint8类型
    # 查找轮廓并筛选掉小区域
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(closed_image)
    for contour in contours:
        # 面积筛选
        if cv2.contourArea(contour) <= area_threshold:
            continue
        # 反射率强度筛选
        mask = np.zeros_like(reflectivity)
        cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        max_val = cv2.minMaxLoc(reflectivity, mask=mask)[1]
        if max_val >= 30:  # 原始反射率最大值需≥25
            cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
    # Canny边缘检测
    blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0) # 先进行高斯模糊减少噪声
    edges = cv2.Canny(blurred_image, 50, 150)
    # 补全四周的像素值
    edges[0, :] = blurred_image[0, :]  # 上边缘
    edges[-1, :] = blurred_image[-1, :]  # 下边缘
    edges[:, 0] = blurred_image[:, 0]  # 左边缘
    edges[:, -1] = blurred_image[:, -1]  # 右边缘

    return edges, reflectivity


def visualize_closed_contours(image_path, grouped_closed_coordinates):
    # 读取原始图像
    image = cv2.imread(image_path)
    color_image = cv2.resize(image, (820, 690))

    # 绘制拟合的椭圆和闭合轮廓
    for closed_coordinates in grouped_closed_coordinates:        
        # 绘制闭合轮廓
        closed_coordinates_np = np.array(closed_coordinates, dtype=np.int32)
        cv2.polylines(color_image, [closed_coordinates_np], isClosed=True, color=(0, 0, 255), thickness=2)

    return color_image

def get_contours(binary_image, reflectivity, area_threshold=0):
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

origin_path = "./static/202406050030.png"
gray_path = "./static/202406050030_gray.png"
sizes = [0, 250, 500, 750]
for size in sizes:
    edges, reflectivity = edge_recognition_0(gray_path, size, 30)
    ellipses, contours_list, max_values, avg_values = get_contours(edges, reflectivity)
    color_img = visualize_closed_contours(origin_path, contours_list)
    result_path = f"./static/202406050030_{size}.png"
    cv2.imwrite(result_path, color_img)