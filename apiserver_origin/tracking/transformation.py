import numpy as np

def create_lookup_table(output_table_path):
    """
    生成查表数组，用于像素点坐标到经纬度的快速转换。
    """
    # 经纬度范围
    left, right = 108.505, 117.505
    up, down = 26.0419, 19.0419
    
    # 生成查表数组
    latitudes = np.linspace(up, down, num=690)  # 纬度查表
    longitudes = np.linspace(left, right, num=820)  # 经度查表
    
    # 构造查表数组（每个像素点对应一个 [lat, lon]）
    lookup_table = np.zeros((690, 820, 2), dtype=np.float32)
    for y in range(690):  # 遍历纬度
        for x in range(820):  # 遍历经度
            lookup_table[y, x, 0] = latitudes[y]  # 纬度
            lookup_table[y, x, 1] = longitudes[x]  # 经度
    
    # 保存查表数组到文件
    np.save(output_table_path, lookup_table)
    print(f"Lookup table saved to {output_table_path}")


def get_latlon_from_coordinates(x, y, lookup_table):
    """
    根据坐标 (x, y) 查找对应的经纬度 (lat, lon)
    """
    width, height, _ = lookup_table.shape
    x_int = int(round(x))  # 四舍五入并转为整数
    if x_int >= height:
        x_int = height - 1
    y_int = int(round(y))  # 四舍五入并转为整数
    if y_int >= width:
        y_int = width - 1
    lat = float(lookup_table[y_int, x_int][0])  # 转换为 Python float 类型
    lon = float(lookup_table[y_int, x_int][1])  # 转换为 Python float 类型
    return lat, lon


def convert_outlines_to_latlon(outlines, lookup_table):
    """
    将像素点轮廓坐标转换为对应的经纬度轮廓坐标
    """
    latlon_outlines = []
    for outline in outlines:
        x, y = outline
        lat, lon = get_latlon_from_coordinates(x, y, lookup_table)
        latlon_outlines.append((float(f"{lat:.3f}"), float(f"{lon:.3f}")))
    return latlon_outlines