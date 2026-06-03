import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt, QSize

import os
import sys

# 根据实际路径修改
PTH_PATH = 'result/resnet18_ls_30/resnet18.pth'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def get_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = 26
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# 图片预处理
DATA_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict_one_image(image_path, model_path, device):
    try:
        model = get_model()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])

        model.to(device)
        model.eval()

        img = Image.open(image_path).convert('RGB')
        img_tensor = DATA_TRANSFORM(img)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, 1)

            pred_class_idx = pred_idx.item()
            confidence = conf.item()

        return pred_class_idx, confidence

    except Exception as e:
        # 如果预测失败（例如模型文件路径错误），返回 None 并打印错误
        print(f"预测核心错误: {e}")
        return None, None


class ASLPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL 字母识别器 (PyQt6)")
        self.setGeometry(100, 100, 800, 600)  # 窗口大小 (x, y, w, h)
        self.current_image_path = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.initUI()

    def initUI(self):
        # 1. 顶部：标题和按钮
        header_layout = QHBoxLayout()

        title = QLabel("美国手语字母识别")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.load_button = QPushButton("加载图片并预测")
        self.load_button.setFont(QFont("Arial", 12))
        self.load_button.clicked.connect(self.open_file_dialog)

        header_layout.addWidget(title)
        header_layout.addWidget(self.load_button)
        self.layout.addLayout(header_layout)
        self.layout.addSpacing(10)

        # 2. 中间：图片显示区域
        self.image_label = QLabel("点击 '加载图片并预测' 上传手语图片")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(QSize(400, 400))  # 限制图片显示区域大小
        self.image_label.setStyleSheet("border: 2px dashed gray;")
        self.layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 3. 底部：结果显示区域
        self.result_label = QLabel("等待预测...")
        self.result_label.setFont(QFont("Arial", 14))
        self.result_label.setStyleSheet("padding: 10px; background-color: #e0f7fa;")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.result_label)

    def open_file_dialog(self):
        """打开文件选择对话框，并触发预测"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png)"  # 过滤器
        )

        if file_name:
            self.current_image_path = file_name
            self.display_image(file_name)
            self.run_prediction(file_name)

    def display_image(self, path):
        """在 QLabel 中显示图片并缩放"""
        pixmap = QPixmap(path)

        # 缩放图片以适应 QLabel 的大小
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setText("")  # 清除文本

    def run_prediction(self, path):
        """执行预测逻辑并更新结果标签"""
        self.result_label.setText("正在预测，请稍候...")

        idx, conf = predict_one_image(path, PTH_PATH, DEVICE)

        if idx is not None and conf is not None:
            predicted_class = CLASS_NAMES[idx]
            confidence_percent = f"{conf:.2%}"

            result_text = f"预测类别: {predicted_class} | 置信度: {confidence_percent}"
            self.result_label.setStyleSheet("background-color: #ccffcc; color: black;")  # 预测成功，绿色背景
            self.result_label.setText(result_text)
        else:
            self.result_label.setStyleSheet("background-color: #ffcccc; color: black;")  # 预测失败，红色背景
            self.result_label.setText("预测失败，请检查模型文件路径或控制台错误信息。")


# --- III. 运行程序 ---

if __name__ == '__main__':
    try:
        if not os.path.exists(PTH_PATH):
            print("-" * 50)
            print(f"错误：模型文件未找到，请检查 PTH_PATH 是否正确：\n{PTH_PATH}")
            print("-" * 50)
            sys.exit(1)

        app = QApplication(sys.argv)
        ex = ASLPredictorApp()
        ex.show()
        sys.exit(app.exec())

    except ImportError as e:
        print("-------------------------------------------------------")
        print(f"导入错误: {e}")
        print("请确认你安装了 PyQt6: pip install pyqt6")
        print("-------------------------------------------------------")
    except Exception as e:
        print(f"发生未知错误: {e}")
        # 打印详细的 traceback 帮助调试
        import traceback

        traceback.print_exc()
        sys.exit(1)