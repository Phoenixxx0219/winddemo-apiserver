def add_span_data(entity, span_time, index, max_value, avg_value, outline, lat, lon, x, y):
    entity["spanData"].append({
        "time": span_time.strftime('%Y-%m-%d %H:%M:%S'),
        "index": index,
        "maxValue": float(f"{max_value:.3f}"),
        "avgValue": float(f"{avg_value:.3f}"),
        "outline": outline,
        "lat": float(f"{lat:.3f}"),
        "lon": float(f"{lon:.3f}"),
        "x": float(f"{x:.3f}"),
        "y": float(f"{y:.3f}"),
        "u": None,
        "v": None,
        "direction": None
    })


def add_entity(entities, id, time, start_time, end_time, start_index, end_index, 
               span_time, index, max_value, avg_value, outline, lat, lon, x, y):
    entity_data = {
        "id": id,
        "time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "startTime": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "endTime": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "startIndex": start_index,
        "endIndex": end_index,
        "speed": 0,
        "direction": None,
        "spanData": []  # 初始化spanData为空
    }
    add_span_data(entity_data, span_time, index, max_value, avg_value, outline, lat, lon, x, y)
    entities.append(entity_data)