from PIL import Image

def combine_images_vertically(image_paths, output_path):
    # 打开所有图像
    images = [Image.open(img_path) for img_path in image_paths]
    
    # 计算新图像的总高度和最大宽度
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    
    # 创建一张新的图像，背景为白色
    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    
    # 拼接图像
    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height
    
    # 保存拼接后的图像
    combined_image.save(output_path)

# 使用示例
image_paths = ['client/4_clients.png', 'client/4_clients.png', 'client/4_clients.png']  # 替换为实际的图片路径
output_path = 'combined_image.png'
combine_images_vertically(image_paths, output_path)
