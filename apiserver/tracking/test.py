# 测试文件，用于本地生成单体识别轮廓图

import cv2
import numpy as np
from datetime import datetime, timedelta

from tracking.recognition import get_area_threshold, edge_recognition_0, edge_recognition_4, edge_recognition_16
from tracking.tracking_func import get_ellipses_and_contours

def batch_process(date, algorithm, size, poolingScale, datatype=12, path="/data/MaxPool", interval_minutes=6):
    color_path = "/data/ImageData"
    color_algorithm = "difftrans_deploy_3h"

    if poolingScale == 0:
        path = "/data/Traffic/image"

    pred_path = path + f"/{date[:-4]}/{datatype}/{algorithm}/{date[-4:-2]}-{date[-2:]}"
    pred_color_path = color_path + f"/{date[:-4]}/{datatype}/{color_algorithm}/{date[-4:-2]}-{date[-2:]}"

    if algorithm == 'real':
        pred_path = pred_path[:-6]
        pred_color_path = pred_color_path[:-6]

    date = datetime.strptime(date,"%Y%m%d%H%M") - timedelta(hours = 1)
    start_time = date + timedelta(minutes = interval_minutes)
    all_images = []
    all_color_images = []
    for real_id in range(10):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        if poolingScale == 0:
            img_path = path + f"/{date_str[:-4]}/{datatype}/real/{date_str}.png"
            color_img_path = color_path + f"/{date_str[:-4]}/{datatype}/real/{date_str}.png"
        else:
            img_path = path + f"/{date_str[:-4]}/{datatype}/real/{poolingScale}/{date_str}.png"
            color_img_path = color_path + f"/{date_str[:-4]}/{datatype}/real/{date_str}.png"
        all_images.append(img_path)
        all_color_images.append(color_img_path)
    for pred_id in range(10,40):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        if poolingScale == 0:
            img_path = pred_path + f"/{date_str}.png"
            color_img_path = pred_color_path + f"/{date_str}.png"
        else:
            img_path = pred_path + f"/{poolingScale}/{date_str}.png"
            color_img_path = pred_color_path + f"/{date_str}.png"
        all_images.append(img_path)
        all_color_images.append(color_img_path)

    edge_images = []
    reflectivitys = []
    area_threshold = get_area_threshold(size)
    for img_path in all_images:
        try:
            if poolingScale == 4:
                edge_image, reflectivity = edge_recognition_4(img_path, area_threshold/16)
            elif poolingScale == 16:
                edge_image, reflectivity = edge_recognition_16(img_path, area_threshold/256)
            else:
                edge_image, reflectivity = edge_recognition_0(img_path, area_threshold)
            edge_images.append(edge_image)
            reflectivitys.append(reflectivity)
        except Exception as e:
            # 图片不存在则设置为None
            edge_images.append(None)
            reflectivitys.append(None)

    return edge_images, reflectivitys, start_time, all_color_images


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
        center = (int(ellipse[0][0]), int(ellipse[0][1]))  # 椭圆中心坐标
        cv2.putText(color_image, str(i+1), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return color_image


date = "202408160600"
algorithm = 'radar_difftrans_deploy_3h'
size = 1000
poolingScale = 0
all_images, reflectivitys, start_time, color_images = batch_process(date, algorithm, size, poolingScale)

scale_x = 1.0
scale_y = 1.0

images_combined = []
for i in range(40):
    ellipses, contours, max_values, avg_values= get_ellipses_and_contours(all_images[i], reflectivitys[i], scale_x, scale_y)
    save_image = visualize_ellipses_and_closed_contours(color_images[i], ellipses, contours)
    save_path = f"./out/{i}.png"
    cv2.imwrite(save_path, save_image)
    images_combined.append(save_image)