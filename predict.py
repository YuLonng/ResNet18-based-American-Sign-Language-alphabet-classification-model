import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


# 1. 导入或定义你的模型结构
# 注意：这里必须和训练时的模型定义完全一致！
# 例如：from my_model_file import MyCNN
# 或者：import torchvision.models as models
def get_model():

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = 26
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

# 2. 定义图片预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 定义类别名称
class_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def predict_one_image(image_path, model_path, device):
    model = get_model()
    checkpoint = torch.load(model_path, map_location=device)
    # 关键步骤：因为你保存的是一个字典，权重在 'model' 这个 key 下
    model.load_state_dict(checkpoint['model'])
    # 将模型放入设备并开启评估模式
    model.to(device)
    model.eval()

    img = Image.open(image_path).convert('RGB')
    img = data_transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    # 预测
    with torch.no_grad():  # 推理不需要计算梯度
        output = model(img)

        # 计算概率（如果模型输出没有经过 Softmax）
        probs = F.softmax(output, dim=1)

        # 获取概率最大的类别索引
        conf, pred_idx = torch.max(probs, 1)

        pred_class_idx = pred_idx.item()
        confidence = conf.item()

    return pred_class_idx, confidence


if __name__ == '__main__':
    # 配置
    img_path = 'image2.jpg'  # [修改这里] 你的测试图片路径
    pth_path = 'result/resnet18_ce/resnet18.pth'  # 你的权重文件路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"正在使用 {device} 进行预测...")

    try:
        idx, conf = predict_one_image(img_path, pth_path, device)

        print("-" * 30)
        print(f"预测结果索引: {idx}")
        if idx < len(class_names):
            print(f"预测类别名称: {class_names[idx]}")
        print(f"置信度 (Confidence): {conf:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查：1.模型定义是否匹配 2.图片路径是否正确")