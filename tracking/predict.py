import math

def calculate_angle(lat_weight, lon_weight):
    # 计算从正东到点(lat_weight, lon_weight)的弧度，并将弧度转换为度数
    angle_rad = math.atan2(lat_weight, lon_weight)
    angle_deg = math.degrees(angle_rad)
    # 将角度调整为正北为 0°，顺时针为正方向
    final_angle = (90 - angle_deg) % 360
    return final_angle


def getDirection(contours):
    # 初始化计数器
    up_move_count = 0
    down_move_count = 0
    right_move_count = 0
    left_move_count = 0
    # 存储最上、最下、最左、最右的点
    tops = []
    bottoms = []
    rights = []
    lefts = []
    # 存储当前帧的最上、最下、最左、最右的点
    prev_top = None
    prev_bottom = None
    prev_left = None
    prev_right = None

    for contour in contours:
        # 找到每个轮廓的最上、最下、最左、最右点
        top = max(contour, key=lambda p: p[0])
        bottom = min(contour, key=lambda p: p[0])
        right = max(contour, key=lambda p: p[1])
        left = min(contour, key=lambda p: p[1])

        tops.append(top[0])
        bottoms.append(bottom[0])
        rights.append(right[1])
        lefts.append(left[1])

        # 计算垂直和水平方向的移动
        if prev_top is not None:  # 确保前一帧存在
            if top[0] > prev_top[0]:  # 上移
                up_move_count += 1
            elif top[0] < prev_top[0]:  # 下移
                down_move_count += 1
            
            if bottom[0] > prev_bottom[0]:  # 上移
                up_move_count += 1
            elif bottom[0] < prev_bottom[0]:  # 下移
                down_move_count += 1
                
            if left[1] < prev_left[1]:  # 左移
                left_move_count += 1
            elif left[1] > prev_left[1]:  # 右移
                right_move_count += 1
            
            if right[1] < prev_right[1]:  # 左移
                left_move_count += 1
            elif right[1] > prev_right[1]:  # 右移
                right_move_count += 1
        
        # 更新前一帧的最上、最下、最左、最右点
        prev_left = left
        prev_right = right
        prev_top = top
        prev_bottom = bottom

    # 计算总的移动次数
    total_lat_moves = up_move_count + down_move_count
    total_lon_moves = left_move_count + right_move_count
    # 计算上下方向的权重比例
    if total_lat_moves > 0:
        if up_move_count > down_move_count:
            lat_weight = up_move_count / total_lat_moves
        elif up_move_count < down_move_count: 
            lat_weight = (0 - down_move_count) / total_lat_moves
        else: 
            lat_weight = 0
    else:
        lat_weight = 0  # 如果没有垂直移动，权重为0
    # 计算左右方向的权重比例
    if total_lon_moves > 0:
        if right_move_count > left_move_count:
            lon_weight = right_move_count / total_lon_moves
        elif right_move_count < left_move_count:
            lon_weight = (0 - left_move_count) / total_lon_moves
        else:
            lon_weight = 0
    else:
        lon_weight = 0  # 如果没有水平移动，权重为0

    # 计算最终的方向向量
    vector = [lat_weight, lon_weight]
    # 计算方向角度，基于四个方向的移动
    if lat_weight == 0 and lon_weight == 0:
        angle = 0  # 如果没有移动，则方向角度为0
    else:
        angle = calculate_angle(vector[0],vector[1])

    return angle, tops, bottoms, rights, lefts, lat_weight, lon_weight


# 计算移动速度
def calculate_speed(move_data, isLat):
    if len(move_data) < 2:
        return 0  # 如果没有足够的数据，返回速度为 0
    # 计算位移（取最后点和起始点的坐标差）
    delta_position = move_data[-1][0] - move_data[0][0]
    interval = (move_data[-1][1] - move_data[0][1]) * 6  # 总时间（分钟）
    if interval == 0:
        return 0  # 防止除以 0

    R = 6371 * 1000  # 地球半径（单位：米）
    # 纬度
    if isLat:
        lat_distance = delta_position * (R * math.pi / 180)
        speed = lat_distance / (interval * 60)  # 纬度方向速度（单位：m/s）
    else:
        lat_resolution = 0.010144927536231883  # 纬度分辨率 (度/像素)
        lat1 = math.radians(move_data[0][0] * lat_resolution)  # 转换为弧度
        lat2 = math.radians(move_data[-1][0] * lat_resolution)  # 转换为弧度
        lon_distance = delta_position * (R * math.pi / 180) * math.cos((lat1 + lat2) / 2)
        speed = lon_distance / (interval * 60)  # 经度方向速度（单位：m/s）
    
    return speed * 3.6


def find_longest_continuous_segment(weight, datas, times):
    longest_segment = []
    current_segment = []

    if weight > 0:
        # 添加起始点
        current_segment.append((datas[0], times[0]))
        # 找出连续移动的坐标
        for i in range(1, len(datas)):
            if datas[i] >= datas[i-1] and datas[i]-datas[i-1] < 0.1:    # 0.1阈值排除合并分裂情况
                current_segment.append((datas[i], times[i]))  # 存储坐标和时间的元组
            else:
                if len(current_segment) > len(longest_segment):
                    longest_segment = current_segment
                current_segment = [(datas[i], times[i])]
        if len(current_segment) > len(longest_segment):
            longest_segment = current_segment  # 最后更新最长段
    elif weight < 0:
        # 添加起始点
        current_segment.append((datas[0], times[0]))
        # 找出连续移动的坐标
        for i in range(1, len(datas)):
            if datas[i] <= datas[i-1] and datas[i-1]-datas[i] < 0.1:    # 0.1阈值排除合并分裂情况
                current_segment.append((datas[i], times[i]))  # 存储坐标和时间的元组
            else:
                if len(current_segment) > len(longest_segment):
                    longest_segment = current_segment
                current_segment = [(datas[i], times[i])]
        if len(current_segment) > len(longest_segment):
            longest_segment = current_segment  # 最后更新最长段

    return longest_segment


def getSpeed(times, tops, bottoms, rights, lefts, lat_weight, lon_weight):
    # 检查times, tops, bottoms, rights, lefts长度是否一致
    if not (len(times) == len(tops) == len(bottoms) == len(rights) == len(lefts)):
        raise ValueError("times, tops, bottoms, rights, lefts must have the same length.")
    
    # 初始化向上、向下、向右、向左的连续坐标数组
    up_move = find_longest_continuous_segment(lat_weight, tops, times)
    bottom_move = find_longest_continuous_segment(lat_weight, bottoms, times)
    right_move = find_longest_continuous_segment(lon_weight, rights, times)
    left_move = find_longest_continuous_segment(lon_weight, lefts, times)

    # 计算纵向和横向的速度
    v_lat = (calculate_speed(up_move, True) + calculate_speed(bottom_move, True)) / 2 if lat_weight != 0 else 0
    v_lon = (calculate_speed(right_move, False) + calculate_speed(left_move, False)) / 2 if lon_weight != 0 else 0
    speed = math.sqrt(v_lat**2 + v_lon**2)

    # 返回纵向和横向的速度
    return v_lat, v_lon, speed


# def getSpeed(ellipse1, ellipse2, interval):
#     """
#     计算椭圆间的速度分量。

#     参数：
#         ellipse1 (tuple): 第一个椭圆的中心点 (x1, y1)，像素坐标。
#         ellipse2 (tuple): 第二个椭圆的中心点 (x2, y2)，像素坐标。
#         interval (float): 时间间隔（分钟）。
#     返回：
#         tuple: (speed, u, v) 总速度大小以及两个方向的速度分量，单位为 m/s。
#     """
#     # 像素坐标
#     x1, y1 = ellipse1
#     x2, y2 = ellipse2
#     # 经纬度分辨率
#     lat_resolution = 0.010144927536231883  # 纬度分辨率 (度/像素)
#     lon_resolution = 0.0109756097560976  # 经度分辨率 (度/像素)
#     # 计算经纬度的实际差值（单位：度）
#     delta_lat = (y2 - y1) * lat_resolution  # 纬度差值
#     delta_lon = (x2 - x1) * lon_resolution  # 经度差值
#     # 将经纬度差值转换为实际的米（距离）
#     lat1 = math.radians(y1 * lat_resolution)  # 转换为弧度
#     lon1 = math.radians(x1 * lon_resolution)  # 转换为弧度
#     lat2 = math.radians(y2 * lat_resolution)  # 转换为弧度
#     lon2 = math.radians(x2 * lon_resolution)  # 转换为弧度
#     # 计算纬度方向的实际距离（单位：米）
#     R = 6371 * 1000  # 地球半径（单位：米）
#     lat_distance = delta_lat * (R * math.pi / 180)
#     # 计算经度方向的实际距离（单位：米），需要乘以纬度余弦因子
#     lon_distance = delta_lon * (R * math.pi / 180) * math.cos((lat1 + lat2) / 2)
#     # Haversine 距离计算总距离
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     total_distance = R * c  # 总距离，单位：米
#     # 速度计算
#     speed = total_distance / (interval * 60)  # 速度，单位：米/秒
#     # 计算速度分量
#     u = lon_distance / (interval * 60)  # 经度方向速度（单位：m/s）
#     v = lat_distance / (interval * 60)  # 纬度方向速度（单位：m/s）

#     return speed, u, v


# def calculate_angle(x, y):
#     """
#     计算向量的角度。

#     参数：
#         x (float): 向量的x分量。
#         y (float): 向量的y分量。

#     返回：
#         float: 角度（0到360度）。
#     """
#     # 计算弧度
#     angle_rad = math.atan2(y, x)
#     # 将弧度转换为度，并确保角度范围在 0 到 360 度之间
#     angle_deg = math.degrees(angle_rad) + 90
#     if angle_deg < 0:
#         angle_deg += 360
#     return angle_deg


# def getDirection(ellipse1, ellipse2):
#     """
#     计算两个椭圆中心的方向向量及角度。

#     参数：
#         ellipse1 (tuple): 第一个椭圆的中心点。
#         ellipse2 (tuple): 第二个椭圆的中心点。
#     返回：
#         tuple: 
#             - angle (float): 方向角度（0到360度）。
#             - vector (list): 方向向量 [dx, dy]。
#     """
#     # 像素坐标
#     x1, y1 = ellipse1
#     x2, y2 = ellipse2
#     # 计算向量
#     vector=[x2-x1,y2-y1]
#     # 计算相对于(x1,y1)的夹角，
#     angle=calculate_angle(vector[0],vector[1])
#     return angle, vector