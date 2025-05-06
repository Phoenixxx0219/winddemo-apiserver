import cv2
import numpy as np

def convert_ellipse_to_original_resolution(ellipse, scale_x, scale_y):
    """
    将池化图上的椭圆参数转换为原分辨率的椭圆参数。
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return ellipse
    # 解构椭圆参数
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    # 转换中心点和轴长
    cx_original = cx * scale_x
    cy_original = cy * scale_y
    major_axis_original = major_axis * scale_x
    minor_axis_original = minor_axis * scale_y
    # 角度保持不变
    return ((cx_original, cy_original), (major_axis_original, minor_axis_original), angle)


def convert_contour_to_original(contour, scale_x, scale_y):
    """
    将池化图上的轮廓转换为原分辨率
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return contour
    
    original_contour = []
    num_points = len(contour)
    for i in range(num_points):
        x, y = contour[i]
        # 放大到原图分辨率
        x_scaled, y_scaled = x * scale_x, y * scale_y
        original_contour.append((x_scaled, y_scaled))
    return original_contour

def process_ellipse_and_contour(ellipse, contour, binary_image, reflectivity, 
                                scale_x, scale_y, ellipses, contours_list, max_values, avg_values):
    """
    处理椭圆和轮廓，计算最大值、平均值，并将结果添加到对应的列表中。
    """
    # 转换椭圆到原始分辨率
    ellipses.append(convert_ellipse_to_original_resolution(ellipse, scale_x, scale_y))
    # 转换轮廓为二维数组并闭合
    contour_2d = contour[:, 0, :]
    closed_contour = np.vstack([contour_2d, contour_2d[0]])
    contour = convert_contour_to_original(closed_contour.tolist(), scale_x, scale_y)
    contours_list.append(contour)

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


def get_ellipses_and_contours(binary_image, reflectivity, scale_x, scale_y, area_threshold=15):
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
                    process_ellipse_and_contour(ellipse, contour, binary_image, reflectivity, scale_x, scale_y, ellipses, contours_list, max_values, avg_values)
                else:
                    # 超出范围，将椭圆中心限制为图片的边界值
                    center_x = np.clip(center_x, 0, width - 1)
                    center_y = np.clip(center_y, 0, height - 1)
                    ellipse = ((center_x, center_y), (major_axis, minor_axis), angle)
                    process_ellipse_and_contour(ellipse, contour, binary_image, reflectivity, scale_x, scale_y, ellipses, contours_list, max_values, avg_values)

    return ellipses, contours_list, max_values, avg_values

def calculate_contour_area(contour):
    """
    计算闭合轮廓的面积
    """
    area = cv2.contourArea(np.array(contour, dtype=np.float32))
    return area

def bbox_intersect(bbox1, bbox2):
    """
    判断两个边界框是否相交。
    
    参数:
        bbox1 (tuple): 第一个边界框的坐标 (x1, y1, x2, y2)。
        bbox2 (tuple): 第二个边界框的坐标 (x1, y1, x2, y2)。
    
    返回:
        bool: 如果两个边界框相交返回 True，否则返回 False。
    """
    ax1, ay1, ax2, ay2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return x1 < x2 and y1 < y2

def calculate_contour_area_overlap_bbox(mask1, mask2, area1, area2, bbox1, bbox2):
    """
    计算两个轮廓掩膜的交并比 (IoU)。
    
    参数:
        mask1 (numpy.ndarray): 第一个轮廓的掩膜。
        mask2 (numpy.ndarray): 第二个轮廓的掩膜。
        area1 (int): 第一个轮廓的面积。
        area2 (int): 第二个轮廓的面积。
        bbox1 (tuple): 第一个轮廓的边界框坐标 (x1, y1, x2, y2)。
        bbox2 (tuple): 第二个轮廓的边界框坐标 (x1, y1, x2, y2)。
    
    返回:
        float: 两个轮廓的交并比 (IoU)。
    """
    ax1, ay1, ax2, ay2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x1 >= x2 or y1 >= y2:
        return 0.0
    mask1_roi = mask1[y1:y2, x1:x2]
    mask2_roi = mask2[y1:y2, x1:x2]
    intersection = cv2.bitwise_and(mask1_roi, mask2_roi)
    intersection_area = cv2.countNonZero(intersection)
    min_area = min(area1, area2)
    return intersection_area / min_area if min_area > 0 else 0.0

def calculate_contour_area_overlap(img_shape, contour1, contour2):
    """
    计算两个闭合轮廓的交并比（IoU）。
    Args:
        img: 图像，用于创建掩膜。
        contour1, contour2: 闭合轮廓的坐标列表。
    Returns:
        IoU: 两个闭合轮廓的交并比。
    """
    # 创建空白图像作为掩膜
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    # 在掩膜上绘制闭合轮廓
    cv2.drawContours(mask1, [np.array(contour1, dtype=np.int32)], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [np.array(contour2, dtype=np.int32)], -1, 255, thickness=cv2.FILLED)
    # 计算交集区域的掩膜
    intersection_mask = cv2.bitwise_and(mask1, mask2)
    intersection_area = np.sum(intersection_mask > 0)
    # 计算每个轮廓的面积
    area1 = calculate_contour_area(contour1)
    area2 = calculate_contour_area(contour2)
    # 计算交并比
    min_area = min(area1, area2)
    iou = intersection_area / min_area if min_area > 0 else 0
    return iou

def determine_ellipse_relationships(img_shape, contours_list1, contours_list2, threshold=0.4):
    """
    判断两个帧之间显著性区块的对应关系，包括“生成”、“延续”、“消散”、“分裂”和“合并”。
    
    参数:
        img_shape (tuple): 图像的形状 (height, width)。
        contours_list1 (list): 第一帧的轮廓列表。
        contours_list2 (list): 第二帧的轮廓列表。
        threshold (float): 交并比的阈值，默认值为 0.4。
    
    返回:
        dict: 包含两个帧之间轮廓关系的字典。
    """
    # 预处理第一帧的轮廓
    mask1_list, area1_list, bbox1_list = [], [], []
    for contour in contours_list1:
        contour_np = np.array(contour, dtype=np.int32)  # 将轮廓转换为 NumPy 数组
        mask = np.zeros(img_shape, dtype=np.uint8)  # 创建空白掩膜
        cv2.drawContours(mask, [contour_np], -1, 255, cv2.FILLED)  # 绘制轮廓到掩膜
        mask1_list.append(mask)  # 保存掩膜
        area1_list.append(cv2.countNonZero(mask))  # 计算轮廓面积并保存
        x, y, w, h = cv2.boundingRect(contour_np)  # 计算轮廓的边界框
        bbox1_list.append((x, y, x + w, y + h))  # 保存边界框坐标
    
    # 预处理第二帧的轮廓
    mask2_list, area2_list, bbox2_list = [], [], []
    for contour in contours_list2:
        contour_np = np.array(contour, dtype=np.int32)  # 将轮廓转换为 NumPy 数组
        mask = np.zeros(img_shape, dtype=np.uint8)  # 创建空白掩膜
        cv2.drawContours(mask, [contour_np], -1, 255, cv2.FILLED)  # 绘制轮廓到掩膜
        mask2_list.append(mask)  # 保存掩膜
        area2_list.append(cv2.countNonZero(mask))  # 计算轮廓面积并保存
        x, y, w, h = cv2.boundingRect(contour_np)  # 计算轮廓的边界框
        bbox2_list.append((x, y, x + w, y + h))  # 保存边界框坐标
    
    # 初始化关系字典
    relationships = {"frame1_to_frame2": [], "frame2_to_frame1": []}
    
    # 遍历第一帧的轮廓，检查与第二帧轮廓的对应关系
    for i, (mask1, area1, bbox1) in enumerate(zip(mask1_list, area1_list, bbox1_list)):
        matches = []
        for j, (mask2, area2, bbox2) in enumerate(zip(mask2_list, area2_list, bbox2_list)):
            if not bbox_intersect(bbox1, bbox2):  # 如果边界框不相交，跳过
                continue
            iou = calculate_contour_area_overlap_bbox(mask1, mask2, area1, area2, bbox1, bbox2)  # 计算交并比
            if iou > threshold:  # 如果交并比超过阈值，记录匹配
                matches.append(j)
        # 根据匹配结果确定状态
        state = "消散" if not matches else "分裂" if len(matches) > 1 else "延续"
        relationships["frame1_to_frame2"].append({"frame1_id": i, "frame2_ids": matches, "state": state})
    
    # 遍历第二帧的轮廓，检查与第一帧轮廓的对应关系
    for j, (mask2, area2, bbox2) in enumerate(zip(mask2_list, area2_list, bbox2_list)):
        matches = []
        for i, (mask1, area1, bbox1) in enumerate(zip(mask1_list, area1_list, bbox1_list)):
            if not bbox_intersect(bbox1, bbox2):  # 如果边界框不相交，跳过
                continue
            iou = calculate_contour_area_overlap_bbox(mask1, mask2, area1, area2, bbox1, bbox2)  # 计算交并比
            if iou > threshold:  # 如果交并比超过阈值，记录匹配
                matches.append(i)
        # 根据匹配结果确定状态
        state = "生成" if not matches else "合并" if len(matches) > 1 else "延续"
        relationships["frame2_to_frame1"].append({"frame2_id": j, "frame1_ids": matches, "state": state})
    
    return relationships