>  本项目采用了深度残差网络的设计，主要参考了He等人（2016）发表的ResNet开创性论文 。项目所使用的“American Sign Language Letters”数据集来源于Kaggle开源平台。
# 下载链接
以下是数据集以及一个已经训练好的模型参数的下载链接。

数据集链接：https://www.kaggle.com/datasets/ammarnassanalhajali/american-sign-language-letters

模型参数链接：https://drive.google.com/file/d/1k2Hmay4THh_zURHcSQJ6AQppCM4n-EMp/view?usp=drive_link

# ASL 手语字母识别模型训练指南

这是一个基于 PyTorch 框架构建的深度学习图像分类训练脚本，专门用于识别美国手语（ASL）的26个英文字母。该脚本支持多种主流的卷积神经网络（CNN）架构、不同的损失函数，并集成了完整的数据增强、模型评估及结果可视化功能。



## 环境依赖

在运行代码之前，请确保您的环境中安装了以下 Python 库：

```bash
pip install torch torchvision tqdm matplotlib seaborn numpy scikit-learn

```



## 数据集准备

脚本默认使用 COCO 格式的 JSON 标注文件，并且数据集需要按 `train`、`valid`、`test` 进行划分。请确保您的数据集目录结构如下：

```text
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── ...

```

> **提示**: 如果您的数据集不在默认的 `dataset` 文件夹中，可以在运行时通过 `--data_root` 参数指定实际路径。

## 运行方式与参数说明

您可以通过命令行参数高度自定义训练过程。

### 命令示例

**1. 快速开始（使用默认参数训练 ResNet18）**

```bash
python train.py

```

**2. 更改模型和损失函数**
使用 `efficientnet_v2` 模型搭配 `focal` loss 进行训练，设置批次大小为 32：

```bash
python train.py --model efficientnet_v2 --loss focal --batch_size 32 --epochs 100

```

**3. 断点续训**
如果训练意外中断，或者想在已有模型基础上继续训练：

```bash
python train.py --resume --model_path result/resnet18_ce_50/resnet18.pth

```


## 输出结果说明

训练开始后，脚本会在项目根目录下自动创建 `./result/<模型名>_<损失函数>_<Epoch数>/` 文件夹。训练结束后，该目录下会包含以下文件：

* **`*.pth`**: 验证集上表现最好的模型权重文件。
* **`acc_curves.png` / `loss_curves.png**`: 训练与验证过程的准确率和损失曲线图。
* **`confusion_matrix.png`**: 测试集上的 26 类别混淆矩阵热力图。
* **`predictions_grid.png`**: 16 张测试集图片的预测抽样图（绿色为预测正确，红色为错误，带有置信度）。
* **`classification_metrics.json`**: 包含每个字母的 Precision, Recall, F1-score 的详尽分类报告。
* **`training_report.json`**: 整体训练摘要，包含运行时间、最佳准确率及模型参数量等核心指标。

# 可视化系统
## 🛠️ 环境依赖

在运行此应用程序前，请确保您的 Python 环境中安装了以下核心依赖库。

```bash
pip install torch torchvision Pillow PyQt6

```

## 📂 模型文件配置

该脚本在顶部硬编码了一个默认的模型权重路径。在首次启动应用之前，您必须确保该路径下存在有效的 `.pth` 文件：

* **默认模型路径**: `result/resnet18_ls_30/resnet18.pth` 

**修改路径**

如果模型存放在其他位置，请在代码文件的顶部区域修改 `PTH_PATH` 变量 ：

```python
# 将此处的路径替换为您真实的模型文件路径
PTH_PATH = 'your_custom_model_dir/best_model.pth'

```

## 🚀 运行方式

将上述代码保存为 `app.py`（或类似名称），在终端中执行以下命令启动应用：

```bash
python predict_visualisation.py

```

**操作步骤：**

1. 应用启动后，请稍等一会儿，启动需要一些时间才能加载出页面。

随后点击右上角的“加载图片并预测”
<img src="doc\home_page.png">

2. 加载完图片，系统就会自动对图片当中的手势进行预测。
<img src="doc\after_predict.png">



