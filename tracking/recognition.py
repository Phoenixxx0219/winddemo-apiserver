import cv2
import math
import numpy as np

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
    # 二值化处理，将低于30dBZ的反射率置为0，其他置为255
    _, binary_image = cv2.threshold(reflectivity, reflectivity_threshold, 255, cv2.THRESH_BINARY)
    # 转换为uint8类型
    closed_image = np.uint8(binary_image)
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
        if max_val >= 25:
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

# def edge_recognition_4(image_path, area_threshold, reflectivity_threshold):
#     """
#     基于形态学操作的显著性区块分割与边缘识别，阈值为1（分辨率为206×173）
#     """
#     # 读入图像，像素值为雷达反射率
#     reflectivity = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # 二值化处理，将低于20dBZ的反射率置为0，其他置为255
#     _, binary_image = cv2.threshold(reflectivity, reflectivity_threshold, 255, cv2.THRESH_BINARY)
#     # 1. 形态学闭操作，连接相邻高回波区域
#     kernel = np.ones((3, 3), np.uint8)
#     closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
#     # 2. 去除小的高回波区域，面积阈值为30
#     closed_image = np.uint8(closed_image)  # 转换为uint8类型
#     # 查找轮廓并筛选掉小区域
#     contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_image = np.zeros_like(closed_image)
#     for contour in contours:
#         if cv2.contourArea(contour) > 125:  # 只有面积大于125的显著性区块
#             cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
#     # 3. Canny边缘检测
#     blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0) # 先进行高斯模糊减少噪声
#     edges = cv2.Canny(blurred_image, 50, 150)
#     # 将原图边缘覆盖至结果图中
#     edges[0, :] = blurred_image[0, :]  # 上边缘
#     edges[-1, :] = blurred_image[-1, :]  # 下边缘
#     edges[:, 0] = blurred_image[:, 0]  # 左边缘
#     edges[:, -1] = blurred_image[:, -1]  # 右边缘

#     return edges, reflectivity


# def edge_recognition_16(image_path, area_threshold, reflectivity_threshold):
#     """
#     基于形态学操作的显著性区块分割与边缘识别，阈值为1（分辨率为52×44）
#     """
#     # 读入图像，像素值为雷达反射率
#     reflectivity = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # 1. 二值化处理，将低于20dBZ的反射率置为0，其他置为255
#     _, binary_image = cv2.threshold(reflectivity, reflectivity_threshold, 255, cv2.THRESH_BINARY)
#     # 2. 去除小的高回波区域，面积阈值为2
#     closed_image = np.uint8(binary_image)  # 转换为uint8类型
#     # 查找轮廓并筛选掉小区域
#     contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_image = np.zeros_like(closed_image)
#     for contour in contours:
#         if cv2.contourArea(contour) > 35:  # 只有面积大于35的显著性区块
#             cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
#     # 3. Canny边缘检测
#     blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0) # 先进行高斯模糊减少噪声
#     edges = cv2.Canny(blurred_image, 50, 150)
#     # 将原图边缘覆盖至结果图中
#     edges[0, :] = blurred_image[0, :]  # 上边缘
#     edges[-1, :] = blurred_image[-1, :]  # 下边缘
#     edges[:, 0] = blurred_image[:, 0]  # 左边缘
#     edges[:, -1] = blurred_image[:, -1]  # 右边缘

#     return edges, reflectivity


# def edge_recognition(image_path):
#     """
#     基于形态学操作的显著性区块分割与边缘识别，阈值为30
#     """
#     # 读入图像，像素值为雷达反射率
#     reflectivity = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # 二值化处理，将低于30dBZ的反射率置为0，其他置为255
#     _, binary_image = cv2.threshold(reflectivity, 30, 255, cv2.THRESH_BINARY)
#     # 1. 形态学闭操作，连接相邻高回波区域
#     kernel = np.ones((7, 7), np.uint8)
#     closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
#     # 2. 去除小的高回波区域，面积阈值为500
#     closed_image = np.uint8(closed_image)  # 转换为uint8类型
#     # 查找轮廓并筛选掉小区域
#     contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_image = np.zeros_like(closed_image)
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # 只有面积大于500的显著性区块
#             cv2.drawContours(filtered_image, [contour], -1, (255), thickness=cv2.FILLED)
#     # 3. Canny边缘检测
#     blurred_image = cv2.GaussianBlur(filtered_image, (3, 3), 0) # 先进行高斯模糊减少噪声
#     edges = cv2.Canny(blurred_image, 50, 150)
#     # 4. 补全四周的像素值
#     edges[0, :] = blurred_image[0, :]
#     edges[-1, :] = blurred_image[-1, :]
#     edges[:, 0] = blurred_image[:, 0]
#     edges[:, -1] = blurred_image[:, -1]
#
#     return edges, reflectivity