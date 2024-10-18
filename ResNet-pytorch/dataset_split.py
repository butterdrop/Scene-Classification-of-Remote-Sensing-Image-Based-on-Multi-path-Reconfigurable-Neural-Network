import os
import shutil
import random

input_folder = r'C:\Users\Lan\PycharmProjects\Major\ResNet-pytorch\dataset\SIRI_WHU'
output_folder = r'images\SIRI-WHU_images'
train_ratio = 0.9
test_ratio = 0.1
image_cls = 200  # 每个类别要随机选取的图像数量

output_folder_train = os.path.join(output_folder, "train")
output_folder_test = os.path.join(output_folder, "test")

# 创建训练和测试集目录
os.makedirs(output_folder_train, exist_ok=True)
os.makedirs(output_folder_test, exist_ok=True)

# 按类处理图像
for root, dirs, files in os.walk(input_folder):
    image_list = []  # 存储当前类别的图像路径

    for filename in files:
        if filename.endswith(".jpg"):
            image_path = os.path.join(root, filename)
            image_list.append(image_path)

    # 如果当前类别的图像数量大于或等于所需数量
    if len(image_list) >= image_cls:
        # 随机选择图像
        selected_images = random.sample(image_list, image_cls)

        # 创建类特定的训练和测试目录
        cls = os.path.basename(root)  # 获取类名
        output_folder_cls_train = os.path.join(output_folder_train, cls)
        output_folder_cls_test = os.path.join(output_folder_test, cls)

        os.makedirs(output_folder_cls_train, exist_ok=True)
        os.makedirs(output_folder_cls_test, exist_ok=True)

        # 拆分图像为训练和测试集
        random.shuffle(selected_images)  # 打乱所选图像列表
        train_image_list = selected_images[:int(image_cls * train_ratio)]
        test_image_list = selected_images[int(image_cls * train_ratio):]

        # 复制图像到相应的文件夹
        for img_path in train_image_list:
            shutil.copy(img_path, output_folder_cls_train)
        for img_path in test_image_list:
            shutil.copy(img_path, output_folder_cls_test)

