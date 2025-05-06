import json
import numpy as np
from datetime import datetime, timedelta
from recognition import get_area_threshold, edge_recognition
from tracking_func import get_ellipses_and_contours, calculate_contour_area, calculate_contour_area_overlap, determine_ellipse_relationships, getSpeedAndDirection
from transformation import convert_outlines_to_latlon, get_latlon_from_coordinates
from predict import linear_regression_direction, getSpeed, getDirection, getSpeed2, getDirection2
from data import add_entity, add_span_data


def batch_process(date, algorithm, size, reflectivity_threshold,
                  datatype=12, path="D:/University/MyForecastApp/winddemo-apiserver/static/Traffic/image", interval_minutes=6):
    """
    读取图片并识别轮廓
    """
    pred_path = path + f"/{date[:-4]}/{datatype}/{algorithm}/{date[-4:-2]}-{date[-2:]}"    
    if algorithm == 'real':
        pred_path = pred_path[:-6]
    date = datetime.strptime(date,"%Y%m%d%H%M") - timedelta(hours = 1)
    start_time = date + timedelta(minutes = interval_minutes)
    all_images = []
    for real_id in range(10):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        img_path = path + f"/{date_str[:-4]}/{datatype}/real/{date_str}.png"
        all_images.append(img_path)
    for pred_id in range(10,40):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        img_path = pred_path + f"/{date_str}.png"
        all_images.append(img_path)

    edge_images = []
    reflectivitys = []
    area_threshold = get_area_threshold(size)
    for img_path in all_images:
        try:
            edge_image, reflectivity = edge_recognition(img_path, area_threshold, reflectivity_threshold)
            edge_images.append(edge_image)
            reflectivitys.append(reflectivity)
        except Exception as e:
            # 图片不存在则设置为None
            edge_images.append(None)
            reflectivitys.append(None)

    return edge_images, reflectivitys, start_time


def monomer_tracking_test(date, algorithm, size=1000, reflectivity_threshold=20, 
                     interval_minutes=6, lookup_table_path="D:/University/MyForecastApp/winddemo-apiserver/static/lookup_table.npy"):
    """
    date: 时间
    algorithm: 预测算法。
    size: 单体面积阈值大小（单位：平方公里）。
    reflectivity_threshold: 单体识别雷达反射率的阈值大小（单位：dBZ）
    interval_minutes: 两帧图像之间的时间间隔（单位：分钟）。
    lookup_table_path: 指定的查表文件路径，用于从像素坐标转换为地理坐标（经纬度）。
    """
    # 算法名字映射
    alg_name_map = {
        'radar_real_3h':'real',
    }
    if algorithm in alg_name_map.keys():
        algorithm = alg_name_map[algorithm]
    # 加载查表数组（lookup table）
    lookup_table = np.load(lookup_table_path)
    # 对文件夹中的图片提取轮廓
    all_images, reflectivitys, start_time = batch_process(date, algorithm, size, reflectivity_threshold)
    cur_time = start_time + timedelta(minutes=9 * interval_minutes)

    entities = []  # 全局编号的实体列表
    entity_mapping = {}  # 当前帧中的ID到全局实体ID的映射
    first_ellipse_info = {} # 实体第一个拟合椭圆的信息
    last_ellipse_info = {}  # 实体最后一个拟合椭圆的信息
    last_outline_info = {}  # 实体最后一个闭合轮廓的信息

    # 检查all_images中是否全为空
    non_empty_images = [image for image in all_images if image is not None]
    if not non_empty_images:  # 如果所有图片都为空
        output_data = {
            "algorithm": algorithm,
            "entities": entities
        }
        return output_data

    # 选取第一张非空的图片并赋值pooled_height, pooled_width
    image = non_empty_images[0]
    pooled_height, pooled_width = image.shape
    # 计算缩放因子
    original_width, original_height = 820, 690
    scale_x = original_width / pooled_width
    scale_y = original_height / pooled_height

    img_shape = (original_height, original_width)

    # 提前计算每帧图像的轮廓和椭圆信息
    ellipses_list = []
    outlines_list = []
    max_values_list = []
    avg_values_list = []
    relation_lists = {}
    for img, reflectivity in zip(all_images, reflectivitys):
        if img is not None:
            ellipses, outlines, max_values, avg_values = get_ellipses_and_contours(img, reflectivity, scale_x, scale_y)
        else:
            ellipses, outlines, max_values, avg_values = [], [], [], []
        ellipses_list.append(ellipses)
        outlines_list.append(outlines)
        max_values_list.append(max_values)
        avg_values_list.append(avg_values)

    for i in range(len(all_images) - 1):
        if all_images[i] is None or all_images[i + 1] is None:
            continue

        ellipses1 = ellipses_list[i]
        outlines1 = outlines_list[i]
        max_values1 = max_values_list[i]
        avg_values1 = avg_values_list[i]
        index1 = i + 1

        ellipses2 = ellipses_list[i + 1]
        outlines2 = outlines_list[i + 1]
        max_values2 = max_values_list[i + 1]
        avg_values2 = avg_values_list[i + 1]
        index2 = i + 2
        
        relationships = determine_ellipse_relationships(img_shape, outlines1, outlines2)

        # 初始化当前帧的实体编号映射
        if i == 0:
            for idx, ellipse in enumerate(ellipses1):
                global_id = len(entities) + 1
                entity_mapping[idx] = global_id
                x, y = ellipse[0]
                lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                outline = convert_outlines_to_latlon(outlines1[idx], lookup_table)
                add_entity(entities, global_id, cur_time, start_time, start_time, index1, index1, 
                           start_time, index1, max_values1[idx], avg_values1[idx], outline, lat, lon, x, y)
                first_ellipse_info[global_id] = ellipse
                last_outline_info[global_id] = outlines1[idx]
                last_ellipse_info[global_id] = ellipse
                relation_lists[global_id] = []

        new_entity_mapping = {}  # 下一帧的实体编号映射

        # img2对应的时间
        img_time = start_time + timedelta(minutes=(i + 1) * interval_minutes)
        # 更新延续状态
        for rel in relationships["frame1_to_frame2"]:
            if rel["state"] == "延续":
                frame1_id = rel["frame1_id"]
                frame2_id = rel["frame2_ids"][0]
                global_id = entity_mapping[frame1_id]

                # 检查 frame2_id 在 frame2_to_frame1 中的状态是否为合并
                frame2_rel = next(
                    (f2_rel for f2_rel in relationships["frame2_to_frame1"] if f2_rel["frame2_id"] == frame2_id),
                    None
                )
                if frame2_rel and frame2_rel["state"] == "合并":
                    # 如果 frame2_id 的状态为合并，跳过更新
                    continue

                if frame2_id not in new_entity_mapping:
                    # 更新数据
                    entities[global_id - 1]["endTime"] = img_time.strftime('%Y-%m-%d %H:%M:%S')
                    entities[global_id - 1]["endIndex"] = index2

                    x, y = ellipses2[frame2_id][0]
                    lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                    outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                    add_span_data(entities[global_id - 1], img_time, index2, max_values2[frame2_id], 
                                  avg_values2[frame2_id], outline, lat, lon, x, y)
                    last_outline_info[global_id] = outlines2[frame2_id] # 更新实体椭圆信息
                    last_ellipse_info[global_id] = ellipses2[frame2_id]
                    new_entity_mapping[frame2_id] = global_id
                    relation_lists[global_id].append((index2, '延续'))
            
            elif rel["state"] == "分裂":
                # 找到面积最大的单体，其延续分裂前单体的信息
                max_area = -1
                max_frame2_id = None
                for frame2_id in rel["frame2_ids"]:
                    area = calculate_contour_area(outlines2[frame2_id])
                    if area > max_area:
                        max_area = area
                        max_frame2_id = frame2_id
                # 延续面积最大的单体
                if max_frame2_id is not None and max_frame2_id not in new_entity_mapping:
                    frame1_id = rel["frame1_id"]
                    global_id = entity_mapping[frame1_id]
                    # 更新数据
                    entities[global_id - 1]["endTime"] = img_time.strftime('%Y-%m-%d %H:%M:%S')
                    entities[global_id - 1]["endIndex"] = index2

                    x, y = ellipses2[max_frame2_id][0]
                    lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                    outline = convert_outlines_to_latlon(outlines2[max_frame2_id], lookup_table)
                    # 更新原实体信息
                    add_span_data(entities[global_id - 1], img_time, index2, max_values2[max_frame2_id], 
                                  avg_values2[max_frame2_id], outline, lat, lon, x, y)
                    relation_lists[global_id].append((index2, '分裂'))
                    last_outline_info[global_id] = outlines2[max_frame2_id]  # 更新椭圆信息
                    last_ellipse_info[global_id] = ellipses2[max_frame2_id]
                    new_entity_mapping[max_frame2_id] = global_id

                for frame2_id in rel["frame2_ids"]:
                    if frame2_id == max_frame2_id and frame2_id in new_entity_mapping:
                        continue
                    # 在此检查当前 frame2_id 是否与之前的实体匹配
                    for global_id, last_info in last_outline_info.items():
                        # 若交并比大于0.4且间隔不超过30min，则认为是延续，而不是生成新实体
                        iou = calculate_contour_area_overlap(img_shape, last_info, outlines2[frame2_id])
                        interval_num = index2 - entities[global_id - 1]["endIndex"]
                        if iou > 0.4 and interval_num <= 5 and global_id != entity_mapping[frame1_id]:
                            # 更新数据
                            entities[global_id - 1]["endTime"] = img_time.strftime('%Y-%m-%d %H:%M:%S')
                            entities[global_id - 1]["endIndex"] = index2

                            x, y = ellipses2[frame2_id][0]
                            lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                            outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                            add_span_data(entities[global_id - 1], img_time, index2, max_values2[frame2_id], 
                                          avg_values2[frame2_id], outline, lat, lon, x, y)

                            relation_lists[global_id].append((index2, '分裂'))
                            last_outline_info[global_id] = outlines2[frame2_id] # 更新实体椭圆信息
                            last_ellipse_info[global_id] = ellipses2[frame2_id]
                            new_entity_mapping[frame2_id] = global_id
                            break
                    # 若不匹配，则认为是新生成的
                    if frame2_id not in new_entity_mapping:
                        new_global_id = len(entities) + 1
                        new_entity_mapping[frame2_id] = new_global_id
                        x, y = ellipses2[frame2_id][0]
                        lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                        outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                        add_entity(entities, new_global_id, cur_time, img_time, img_time, index2, index2, 
                                   img_time, index2, max_values2[frame2_id], avg_values2[frame2_id], outline, lat, lon, x, y)
                        first_ellipse_info[new_global_id] = ellipses2[frame2_id]
                        last_outline_info[new_global_id] = outlines2[frame2_id]
                        last_ellipse_info[new_global_id] = ellipses2[frame2_id]
                        relation_lists[new_global_id] = []
                        relation_lists[new_global_id].append((index2, '分裂'))

        for rel in relationships["frame2_to_frame1"]:
            if rel["state"] == "生成":
                frame2_id = rel["frame2_id"]
                if frame2_id in new_entity_mapping:
                    continue
                # 在此检查当前 frame2_id 是否与之前的实体匹配
                for global_id, last_info in last_outline_info.items():
                    # 若交并比大于0.4且间隔不超过30min，则认为是延续，而不是生成新实体
                    iou = calculate_contour_area_overlap(img_shape, last_info, outlines2[frame2_id])
                    interval_num = index2 - entities[global_id - 1]["endIndex"]
                    if iou > 0.4 and interval_num <= 6:
                        # 更新数据
                        entities[global_id - 1]["endTime"] = img_time.strftime('%Y-%m-%d %H:%M:%S')
                        entities[global_id - 1]["endIndex"] = index2

                        x, y = ellipses2[frame2_id][0]
                        lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                        outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                        add_span_data(entities[global_id - 1], img_time, index2, max_values2[frame2_id], 
                                      avg_values2[frame2_id], outline, lat, lon, x, y)

                        last_outline_info[global_id] = outlines2[frame2_id] # 更新实体椭圆信息
                        last_ellipse_info[global_id] = ellipses2[frame2_id]
                        new_entity_mapping[frame2_id] = global_id
                        relation_lists[global_id].append((index2, '延续'))
                        break
                # 若不匹配，则认为是新生成的
                if frame2_id not in new_entity_mapping:
                    global_id = len(entities) + 1
                    relation_lists[global_id] = []
                    relation_lists[global_id].append((index2, '生成'))
                    new_entity_mapping[frame2_id] = global_id
                    x, y = ellipses2[frame2_id][0]
                    lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                    outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                    add_entity(entities, global_id, cur_time, img_time, img_time, index2, index2, 
                               img_time, index2, max_values2[frame2_id], avg_values2[frame2_id], outline, lat, lon, x, y)
                    first_ellipse_info[global_id] = ellipses2[frame2_id]
                    last_outline_info[global_id] = outlines2[frame2_id]
                    last_ellipse_info[global_id] = ellipses2[frame2_id]
            elif rel["state"] == "合并":
                frame2_id = rel["frame2_id"]

                # 找到面积最大的单体，其延续分裂前单体的信息
                max_area = -1
                max_frame1_id = None
                for frame1_id in rel["frame1_ids"]:
                    area = calculate_contour_area(outlines1[frame1_id])
                    if area > max_area:
                        max_area = area
                        max_frame1_id = frame1_id

                # 延续面积最大的单体
                if max_frame1_id is not None:
                    global_id = entity_mapping[max_frame1_id]
                    relation_lists[global_id].append((index2, '合并'))
                    if frame2_id not in new_entity_mapping:
                        # 更新数据
                        entities[global_id - 1]["endTime"] = img_time.strftime('%Y-%m-%d %H:%M:%S')
                        entities[global_id - 1]["endIndex"] = index2

                        x, y = ellipses2[frame2_id][0]
                        lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                        outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                        # 更新原实体信息
                        add_span_data(entities[global_id - 1], img_time, index2, max_values2[frame2_id], 
                                      avg_values2[frame2_id], outline, lat, lon, x, y)
                        last_outline_info[global_id] = outlines2[frame2_id]  # 更新椭圆信息
                        last_ellipse_info[global_id] = ellipses2[frame2_id]
                        new_entity_mapping[frame2_id] = global_id
                else:
                    if frame2_id not in new_entity_mapping:
                        global_id = len(entities) + 1
                        relation_lists[global_id] = []
                        relation_lists[global_id].append((index2, '合并'))
                        new_entity_mapping[frame2_id] = global_id
                        x, y = ellipses2[frame2_id][0]
                        lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
                        outline = convert_outlines_to_latlon(outlines2[frame2_id], lookup_table)
                        add_entity(entities, global_id, cur_time, img_time, img_time, index2, index2, 
                                   img_time, index2, max_values2[frame2_id], avg_values2[frame2_id], outline, lat, lon, x, y)
                        first_ellipse_info[global_id] = ellipses2[frame2_id]
                        last_outline_info[global_id] = outlines2[frame2_id]
                        last_ellipse_info[global_id] = ellipses2[frame2_id]

        entity_mapping = new_entity_mapping

    for entity in entities:
        span_data_list = entity.get("spanData", [])
        # 如果有至少两个数据点，则尝试线性回归拟合
        if span_data_list and len(span_data_list) > 2:
            # 收集所有中心点数据
            center_points = [(data["x"], data["y"]) for data in span_data_list]
            angle_reg, r2 = linear_regression_direction(center_points)
            # 使用第一帧和最后一帧的中心点计算时间间隔
            time_format = '%Y-%m-%d %H:%M:%S'
            t1 = span_data_list[0]["time"]
            t2 = span_data_list[-1]["time"]
            # 如果时间为字符串，则转换为 datetime 对象
            if isinstance(t1, str):
                t1 = datetime.strptime(t1, time_format)
            if isinstance(t2, str):
                t2 = datetime.strptime(t2, time_format)
            interval = (t2 - t1).total_seconds() / 60.0
            if interval <= 0:
                interval = 1  # 防止除 0 的情况
            # 用第一个和最后一个中心点计算速度
            ellipse1 = center_points[0]
            ellipse2 = center_points[-1]
            speed1, u, v = getSpeed(ellipse1, ellipse2, interval)
            angle1, vector = getDirection(ellipse1, ellipse2)

            # 如果线性回归拟合效果不理想，则采用原来的轮廓连续性方法
            contours = []
            times = []
            for span_data in span_data_list:
                contours.append(span_data["outline"])
                times.append(span_data["index"])
            direction2, tops, bottoms, rights, lefts, lat_weight, lon_weight = getDirection2(contours)
            _, _, speed2 = getSpeed2(times, tops, bottoms, rights, lefts, lat_weight, lon_weight)

            gid = entity['id']
            angle1, speed1, direction2, speed2 = getSpeedAndDirection(date, gid, angle1, speed1, direction2, speed2)

            if r2 >= 0.7:
                entity["direction"] = angle1
                entity["speed"] = speed1
            else:
                entity["direction"] = direction2
                entity["speed"] = speed2

            print("entity id:", entity["id"], "r2:", r2)
            rels = relation_lists.get(gid, [])
            print(f"relations: {rels}")
            print("质心法 ", "speed:", speed1, ", direction:", angle1, )
            print("轮廓法 ", "speed:", speed2, ", direction:", direction2)
        else:
            # 若数据点不足（或 spanData 为空），直接采用轮廓连续性方法
            contours = []
            times = []
            for span_data in span_data_list:
                contours.append(span_data["outline"])
                times.append(span_data["index"])
            direction, tops, bottoms, rights, lefts, lat_weight, lon_weight = getDirection2(contours)
            _, _, speed = getSpeed2(times, tops, bottoms, rights, lefts, lat_weight, lon_weight)
            entity["direction"] = direction
            entity["speed"] = speed       

    output_data = {
        "algorithm": algorithm,
        "entities": entities
    }

    return output_data

