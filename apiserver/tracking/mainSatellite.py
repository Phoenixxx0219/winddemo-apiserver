import json
import numpy as np
from datetime import datetime, timedelta
from tracking.recognitionSatellite import edge_recognition
from tracking.tracking_func import get_ellipses_and_contours, calculate_contour_area, calculate_contour_area_overlap, determine_ellipse_relationships
from tracking.transformation import convert_outlines_to_latlon, get_latlon_from_coordinates
from tracking.predict import getSpeed, getDirection
from tracking.data import add_entity, add_span_data


def batch_process(date, algorithm, datatype=11, path="/data/Traffic/image", interval_minutes=15):

    pred_path = path + f"/{date[:-4]}/{datatype}/{algorithm}/{date[-4:-2]}-{date[-2:]}"    
    if algorithm == 'real':
        pred_path = pred_path[:-6]
    date = datetime.strptime(date,"%Y%m%d%H%M") - timedelta(hours = 2)
    start_time = date + timedelta(minutes = interval_minutes)
    all_images = []
    for real_id in range(8):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        img_path = path + f"/{date_str[:-4]}/{datatype}/real/{date_str}.png"
        all_images.append(img_path)
    for pred_id in range(8,24):
        date = date + timedelta(minutes = interval_minutes)
        date_str = date.strftime("%Y%m%d%H%M")
        img_path = pred_path + f"/{date_str}.png"
        all_images.append(img_path)

    edge_images = []
    reflectivitys = []
    for img_path in all_images:
        try:
            edge_image, reflectivity = edge_recognition(img_path)
            edge_images.append(edge_image)
            reflectivitys.append(reflectivity)
        except Exception as e:
            # 图片不存在则设置为None
            edge_images.append(None)
            reflectivitys.append(None)

    return edge_images, reflectivitys, start_time


def satellite_tracking(date, algorithm, interval_minutes=15, lookup_table_path="/data/cloud_latlon_lookup_table_average.npy"):
    """
    date: 时间
    algorithm (int): 预测算法。
    interval_minutes (int, 默认值：15): 两帧图像之间的时间间隔（单位：分钟）。
    lookup_table_path (str): 指定的查表文件路径，用于从像素坐标转换为地理坐标（经纬度）。
    """
    # 算法名字映射
    # alg_name_map = {
    #     'radar_real_3h':'real',
    # }
    # if algorithm in alg_name_map.keys():
    #     algorithm = alg_name_map[algorithm]
    # 加载查表数组（lookup table）
    lookup_table = np.load(lookup_table_path)
    # 对文件夹中的图片提取轮廓
    all_images, reflectivitys, start_time = batch_process(date, algorithm)
    cur_time = start_time + timedelta(minutes=7 * interval_minutes)

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
    original_width, original_height = 1824, 1060
    scale_x = original_width / pooled_width
    scale_y = original_height / pooled_height

    img_shape = (original_height, original_width)

    for i in range(len(all_images) - 1):
        if(all_images[i] is None or all_images[i+1] is None):
            continue
        img1 = all_images[i]
        reflectivity1 = reflectivitys[i]
        index1 = i + 1
        img2 = all_images[i + 1]
        reflectivity2 = reflectivitys[i + 1]
        index2 = i + 2
        ellipses1, outlines1, max_values1, avg_values1 = get_ellipses_and_contours(img1, reflectivity1, scale_x, scale_y)
        ellipses2, outlines2, max_values2, avg_values2 = get_ellipses_and_contours(img2, reflectivity2, scale_x, scale_y)
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
                if max_frame2_id is not None:
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
                    last_outline_info[global_id] = outlines2[max_frame2_id]  # 更新椭圆信息
                    last_ellipse_info[global_id] = ellipses2[max_frame2_id]
                    new_entity_mapping[max_frame2_id] = global_id

                for frame2_id in rel["frame2_ids"]:
                    if frame2_id == max_frame2_id:
                        continue
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

        for rel in relationships["frame2_to_frame1"]:
            if rel["state"] == "生成":
                frame2_id = rel["frame2_id"]
                # 在此检查当前 frame2_id 是否与之前的实体匹配
                for global_id, last_info in last_outline_info.items():
                    # 若交并比大于0.4且间隔不超过30min，则认为是延续，而不是生成新实体
                    iou = calculate_contour_area_overlap(img_shape, last_info, outlines2[frame2_id])
                    interval_num = index2 - entities[global_id - 1]["endIndex"]
                    if iou > 0.4 and interval_num <= 5:
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
                        break
                # 若不匹配，则认为是新生成的
                if frame2_id not in new_entity_mapping:
                    global_id = len(entities) + 1
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
                    global_id = len(entities) + 1
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
        contours = []
        times = []
        span_data_list = entity.get("spanData", [])
        if span_data_list:
            for span_data in span_data_list:
                outline = span_data['outline']
                contours.append(outline)
                index = span_data['index']
                times.append(index)
        direction, tops, bottoms, rights, lefts, lat_weight, lon_weight = getDirection(contours)
        _, _, speed = getSpeed(times, tops, bottoms, rights, lefts, lat_weight, lon_weight)
        entity["direction"] =  direction 
        entity["speed"] = speed        

    output_data = {
        "algorithm": algorithm,
        "entities": entities
    }

    return output_data


if __name__=='__main__':
    import time
    #记录开始时间
    start_time = time.perf_counter()

    date = "202501122230"
    algorithm = 'cloud_dugs_unet_3h'
    output_data = satellite_tracking(date, algorithm)
    with open("./out1.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    # 记录结束时间
    end_time = time.perf_counter()
    # 计算并打印执行时间
    execution_time = end_time - start_time
    print(f"The test function took {execution_time} seconds to complete.")

    def check_start_time_equality(json_file_path):
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        entities = data.get("entities", [])
        for entity in entities:
            id = entity.get('id')
            start = entity.get('startTime')
            end = entity.get('endTime')
            direction = entity.get('direction')
            speed = entity.get('speed')
            span_data_list = entity.get("spanData", [])
            print("id:", id, ", start:", start, ", end:", end, ", direction:", direction, ", speed:", speed)
            if span_data_list:
                for data in span_data_list:
                    time = data.get('time')
                    direction = data.get('direction')
                    x = data.get('x')
                    y = data.get('y')
                    lat = data.get('lat')
                    lon = data.get('lon')
                    print("time:", time, ", direction:", direction, ", x:", x, ", y:", y, ", lat:", lat, ", lon:", lon)
            else:
                print(f"Entity with id {entity.get('id')} has no spanData.")


    check_start_time_equality("out1.json")