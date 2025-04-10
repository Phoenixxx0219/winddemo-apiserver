import os
import cv2
import numpy as np

# def abstract_color(image_path):
#     """
#     处理图片，将其 HSV 颜色映射到对应的反射率。
#     参数：
#         image_path (str): 输入图片的路径
#     返回值：
#         np.ndarray: 处理后的灰度图，像素值为对应的反射率 * 2
#     """
#     # 定义 HSV 到反射率的映射关系（各映射对可根据实际情况调整）
#     hsv_to_reflection = [
#         ([90, 255, 236], 7),
#         ([101, 254, 246], 12),
#         ([120, 255, 246], 17),
#         ([60, 255, 239], 22),
#         ([60, 255, 200], 27),
#         ([60, 255, 144], 32),
#         ([30, 255, 255], 37),
#         ([25, 255, 231], 42),
#         ([17, 253, 255], 47),
#         ([0, 255, 255], 52),
#         ([0, 255, 166], 57),
#         ([0, 255, 101], 62),
#         ([150, 255, 255], 67),
#         ([138, 147, 201], 72)
#     ]
    
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (820, 690))
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # 初始化一张全 0 的灰度图
#     total_img = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
#     for hsv_val, rflct_rate in hsv_to_reflection:
#         lower = np.array(hsv_val)
#         upper = np.array(hsv_val)
#         mask = cv2.inRange(hsv, lower, upper)
#         filtered_img = cv2.bitwise_and(image, image, mask=mask)
#         gray_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
#         gray_img = gray_img.astype(np.float32)
#         # 对非零区域赋予对应反射率值
#         gray_img[gray_img > 0] = rflct_rate
#         total_img += gray_img
#     return total_img


def abstract_color(image_path):
    """
    处理图片，将其 HSV 颜色映射到对应的反射率。
    对图片中每个像素点的 HSV 值，选择 hsv_to_reflection 中与其最相似的 HSV，并赋值为对应的反射率。
    如果某像素的 HSV 为 [0, 0, 0]，则该像素赋值为 0（黑色）。
    
    参数：
        image_path (str): 输入图片的路径
    返回值：
        np.ndarray: 灰度图，像素值为对应的反射率
    """
    # 定义 HSV 到反射率的映射关系
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
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    # 调整图片尺寸为固定大小
    image = cv2.resize(image, (820, 690))
    # 将 BGR 图片转换到 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 预先构造候选颜色及其反射率的数组，便于后续计算
    hsv_colors = np.array([h for h, _ in hsv_to_reflection], dtype=np.float32)   # [N, 3]
    reflectivities = np.array([r for _, r in hsv_to_reflection], dtype=np.float32)  # [N,]
    
    h, w = hsv.shape[:2]
    hsv_flat = hsv.reshape(-1, 3).astype(np.float32)  # 展平成 (h*w, 3)
    
    # 初始化输出数组，默认所有像素为 0（黑色）
    reflect_flat = np.zeros((hsv_flat.shape[0],), dtype=np.uint8)
    
    # 找出所有非黑色的像素，即 HSV 不全为 0
    black_mask = np.all(hsv_flat == 0, axis=1)  # True 表示该像素为 [0,0,0]
    non_black_idx = np.where(~black_mask)[0]
    
    if non_black_idx.size > 0:
        non_black_pixels = hsv_flat[non_black_idx]  # 取出非黑像素，形状为 (N, 3)
        # 计算每个非黑像素与 hsv_colors 中各个候选颜色之间的欧氏距离
        dists = np.linalg.norm(non_black_pixels[:, None, :] - hsv_colors[None, :, :], axis=2)
        # 对每个像素，找到距离最小的候选颜色索引
        nearest_idx = np.argmin(dists, axis=1)
        # 将对应的反射率值赋给非黑像素
        reflect_flat[non_black_idx] = reflectivities[nearest_idx].astype(np.uint8)
    
    # 重塑为原图尺寸
    reflect_img = reflect_flat.reshape(h, w)
    
    return reflect_img

def process_images():
    # 源目录和目标目录
    input_root = r"./static/ImageData/20241120/12/111"
    output_root = r"./static/Traffic/image/20241120/12/111"
    
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
