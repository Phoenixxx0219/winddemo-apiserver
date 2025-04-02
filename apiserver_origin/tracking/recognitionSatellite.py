import cv2
import numpy as np

def edge_recognition(image_path):
    """
    基于形态学操作的显著性区块分割与边缘识别，阈值为0（卫星数据）
    """
    # 读入图像，像素值为卫星数据
    reflectivity = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化处理，将低于0的反射率置为0，其他置为255
    _, binary_image = cv2.threshold(reflectivity, 0, 255, cv2.THRESH_BINARY)
    # 去除小的高回波区域，面积阈值为1000
    binary_image = np.uint8(binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    processed_image = np.zeros_like(binary_image)
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(processed_image, [contour], -1, (255), thickness=cv2.FILLED)
    # 形态学闭操作，连接相邻高回波区域
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    closed_image = cv2.morphologyEx(processed_image, cv2.MORPH_CLOSE, kernel)
    # 去除小的高回波区域，面积阈值为2000
    closed_image = np.uint8(closed_image)  # 转换为uint8类型
    # 查找轮廓并筛选掉小区域
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_image = np.zeros_like(closed_image)
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # 只有面积大于3000的显著性区块
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