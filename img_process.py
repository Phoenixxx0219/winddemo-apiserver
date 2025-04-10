import os
import cv2
import numpy as np

def abstract_color(image_path):
    """
    处理图片，将其 HSV 颜色映射到对应的反射率。
    参数：
        image_path (str): 输入图片的路径
    返回值：
        np.ndarray: 处理后的灰度图，像素值为对应的反射率 * 2
    """
    # 定义 HSV 到反射率的映射关系（各映射对可根据实际情况调整）
    hsv_to_reflection = [
        ([90, 255, 236], 7),
        ([101, 254, 246], 12),
        ([120, 255, 246], 17),
        ([60, 255, 239], 22),
        ([60, 255, 200], 27),
        ([60, 255, 144], 32),
        ([30, 255, 255], 37),
        ([25, 255, 231], 42),
        ([17, 253, 255], 47),
        ([0, 255, 255], 52),
        ([0, 255, 166], 57),
        ([0, 255, 101], 62),
        ([150, 255, 255], 67),
        ([138, 147, 201], 72)
    ]
    
    # 读取图片及转换至 HSV 颜色空间
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 初始化一张全0的灰度图（浮点型以保存反射率数值）
    total_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    
    for hsv_val, rflct_rate in hsv_to_reflection:
        # 使用相同的 hsv 值作为上下界构造 mask（匹配完全相等的像素值）
        lower = np.array(hsv_val)
        upper = np.array(hsv_val)
        mask = cv2.inRange(hsv, lower, upper)
        filtered_img = cv2.bitwise_and(image, image, mask=mask)
        gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        gray_img = gray_img.astype(np.float32)
        # 对非零区域赋予对应反射率值
        gray_img[gray_img > 0] = rflct_rate
        # 累加每个区域（这里假设不同映射间不会重叠，如有重叠则值会叠加）
        total_img += gray_img
        
    # 为确保保存时数据格式合适，这里将结果*2之后转换为8位无符号整型
    total_img = np.clip(total_img * 2, 0, 255).astype(np.uint8)
    return total_img

def process_images():
    # 源目录和目标目录
    input_root = r"./static/ImageData/20241120/12/forcast"
    output_root = r"./static/Traffic/image/20241120/12/forcast"
    
    # 若目标根目录不存在则创建
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # 遍历源目录下的所有子文件夹（假设直接子文件夹就是各类别）
    for subfolder in os.listdir(input_root):
        input_subdir = os.path.join(input_root, subfolder)
        # 判断是否为文件夹
        if os.path.isdir(input_subdir):
            # 构造对应的输出子文件夹
            output_subdir = os.path.join(output_root, subfolder)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)
            
            # 遍历当前子文件夹下的所有图片
            for filename in os.listdir(input_subdir):
                # 根据文件扩展名判断是否为图片（可根据实际情况增加）
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    input_image_path = os.path.join(input_subdir, filename)
                    try:
                        # 调用 abstract_color 处理图片
                        processed_img = abstract_color(input_image_path)
                    except Exception as e:
                        print(f"处理图片 {input_image_path} 时发生错误：{e}")
                        continue

                    output_image_path = os.path.join(output_subdir, filename)
                    # 保存处理后的图片
                    success = cv2.imwrite(output_image_path, processed_img)
                    if not success:
                        print(f"保存图片 {output_image_path} 失败！")
            print(f"子文件夹 {subfolder} 处理完成。")
    print("所有图片处理完成！")

if __name__ == "__main__":
    process_images()
