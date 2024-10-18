import torch
from model.resnets import resnet18
# 加载.pth文件
model_path = "pths/resnet18.pth"  # 替换为你的.pth文件路径
model_state_dict = torch.load(model_path)

# 创建一个模型实例
model = resnet18()  # 替换为你的模型类

# 加载权重到模型中
model.load_state_dict(model_state_dict)

# 打印模型结构
print(model)

# 打印模型参数
for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print(param)