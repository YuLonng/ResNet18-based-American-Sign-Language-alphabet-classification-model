import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights, vgg11, VGG11_Weights, mobilenet_v2, MobileNet_V2_Weights, efficientnet_v2_s, EfficientNet_V2_S_Weights
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from datetime import datetime
import json
import pathlib
from dataset import ASLDataset

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, x, target):
        logpt = torch.nn.functional.log_softmax(x, dim=-1)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.alpha is not None:
            loss = self.alpha.gather(0, target) * loss
        return loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

#命令行参数
def get_args_parser():
    parser = argparse.ArgumentParser(description='ASL 手语字母识别 - ResNet18 训练脚本')
    parser.add_argument('--data_root', type=str, default='dataset',
                        help='数据集根目录 (默认: dataset)')
    parser.add_argument('--batch_size', type=int, default=64, help='批大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--workers', type=int, default=0,
                        help='推荐 0)')
    parser.add_argument('--model_path', type=str, default='best_resnet18_asl.pth',
                        help='模型保存路径')
    parser.add_argument('--resume', action='store_true', help='是否续训')
    # Loss 函数选择
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'focal', 'ls'],)
    # 模型选择
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'vgg11', 'mobilenet_v2', 'efficientnet_v2'],
                        help='选择 backbone')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--no_cudnn_benchmark', action='store_true')

    return parser

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch:03d} [TRAIN]', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss/(pbar.n+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    pbar.close()
    return running_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device, epoch=None, name="VAL"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    desc = f"{'TEST' if epoch is None else f'Epoch {epoch:03d}'} [{name}]"
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(pbar.n+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    pbar.close()
    return running_loss / len(loader), 100. * correct / total



plt.rc("font",family='MicroSoft YaHei',weight="bold")
# plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    print(args)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据增强
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集
    train_dataset = ASLDataset(
        json_path=f'{args.data_root}/train/_annotations.coco.json',
        images_dir=f'{args.data_root}/train',
        transform=train_transform
    )
    val_dataset = ASLDataset(
        json_path=f'{args.data_root}/valid/_annotations.coco.json',
        images_dir=f'{args.data_root}/valid',
        transform=val_test_transform
    )
    test_dataset = ASLDataset(
        json_path=f'{args.data_root}/test/_annotations.coco.json',
        images_dir=f'{args.data_root}/test',
        transform=val_test_transform
    )

    ALL_LABELS_IDS = list(range(len(train_dataset.label_to_name)))
    sorted_names = [name for id, name in
                    sorted(train_dataset.label_to_name.items())]
    print(ALL_LABELS_IDS)
    print(sorted_names)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")
    print("类别映射:", {v: k for k, v in train_dataset.label_to_name.items()})
    print("-" * 60)

    # 模型选择
    model_dict = {
        'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
        'vgg11': vgg11(weights=VGG11_Weights.IMAGENET1K_V1),
        'mobilenet_v2': mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),
        'efficientnet_v2': efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    }
    model = model_dict[args.model]

    # 统一修改最后一层
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 26)
    elif hasattr(model, 'classifier'):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 26)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, 26)
    model = model.to(DEVICE)

    #Loss选择
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'ls':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    elif args.loss == 'focal':
        criterion = FocalLoss(gamma=args.gamma)
    else:
        raise ValueError("暂不支持该Loss")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0.0
    start_epoch = 0

    # 续训恢复历史记录
    if args.resume and os.path.exists(args.model_path):
        print(f"正在从 {args.model_path} 加载检查点...")
        ckpt = torch.load(args.model_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['acc']
        print(f"续训成功！从 epoch {start_epoch} 开始，历史最佳 acc: {best_acc:.2f}%")

    print("开始训练！")
    print("=" * 80)
    # 输出路径
    model_loss_dir = args.model+ "_" + args.loss + "_" + str(args.epochs)
    output_dir = pathlib.Path("./result") / model_loss_dir
    output_dir.mkdir(parents=True, exist_ok=True)


    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, epoch, "VAL")

        # 记录曲线
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        status = "NEW BEST!" if val_acc > best_acc else ""
        print(f"Epoch {epoch:03d} | Train: {train_acc:6.2f}% | Val: {val_acc:6.2f}% {status}")

        save_path = (output_dir / args.model).with_suffix(".pth")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs,
            }, save_path)
            print(f"Best model saved! Val Acc: {best_acc:.3f}%")

    plt.rc("font", family='MicroSoft YaHei', weight="bold")

    # 1. 绘制 Loss & Accuracy 曲线
    # plt.figure(figsize=(14, 5))
    plt.figure(figsize=(10, 6))


    # print(output_dir)

    # plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', marker='o')
    plt.plot(val_losses, label='验证损失', marker='o')
    plt.title('训练与验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(output_dir / "loss_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plt.subplot(1, 2, 2)
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='训练准确率', marker='o')
    plt.plot(val_accs, label='验证准确率', marker='o')
    plt.title('训练与验证准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(output_dir / "acc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 测试集混淆矩阵 + 预测对比图
    model.eval()
    all_preds = []
    all_labels = []
    sample_images = []
    sample_preds = []
    sample_labels = []
    sample_probs = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            labels_np = labels.cpu().numpy()
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            probs = outputs.softmax(dim=1).max(dim=1)[0].cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels_np)

            # 收集前16张做展示
            if len(sample_images) < 16:
                for j in range(images.size(0)):
                    if len(sample_images) >= 16:
                        break
                    img = images[j].cpu()
                    # 反归一化
                    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                    img = img * std + mean
                    img = img.permute(1, 2, 0).numpy()
                    img = np.clip(img, 0, 1)
                    sample_images.append(img)
                    sample_preds.append(preds[j])
                    sample_labels.append(labels_np[j])
                    sample_probs.append(probs[j])

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                yticklabels=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    plt.title('ASL 手语字母识别 - 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 预测对比图
    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        img = sample_images[i]
        true_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[sample_labels[i]]
        pred_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[sample_preds[i]]
        color = 'green' if sample_preds[i] == sample_labels[i] else 'red'
        plt.imshow(img)
        plt.title(f"真实: {true_char} → 预测: {pred_char}\n置信度: {sample_probs[i]:.3f}",
                  color=color, fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "predictions_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 测试集最终评估
    target_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    NUM_CLASSES = 26
    labels_to_evaluate = list(range(NUM_CLASSES))  # [0, 1, ..., 25]

    full_report = classification_report(
        all_labels, all_preds,
        labels=labels_to_evaluate,
        target_names=target_names,
        output_dict=True,  # 导出为字典以便保存
        digits=3,
        zero_division=0
    )

    # 计算宏平均 (Macro Avg) 和加权平均 (Weighted Avg)
    macro_precision = full_report['macro avg']['precision']
    macro_recall = full_report['macro avg']['recall']
    macro_f1 = full_report['macro avg']['f1-score']

    weighted_precision = full_report['weighted avg']['precision']
    weighted_recall = full_report['weighted avg']['recall']
    weighted_f1 = full_report['weighted avg']['f1-score']

    print("=" * 60)
    print("分类指标详细报告 (TEST SET)")
    print(classification_report(all_labels, all_preds, labels=labels_to_evaluate, target_names=target_names, digits=3))
    print("=" * 60)

    # 保存完整的分类报告（JSON格式）
    report_path = output_dir / "classification_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=4)
    print(f"分类指标报告已保存至: {report_path}")


    _, test_acc = evaluate(model, test_loader, criterion, DEVICE, name="TEST")
    print(f"测试集准确率: {test_acc:.3f}%")
    print(f"最佳验证准确率: {best_acc:.3f}%")

    report = {
        "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总训练轮数": args.epochs,
        "最佳验证准确率": f"{best_acc:.3f}%",
        "测试集准确率": f"{test_acc:.3f}%",
        # 添加宏平均指标
        "测试集-宏平均Precision": f"{macro_precision:.4f}",
        "测试集-宏平均Recall": f"{macro_recall:.4f}",
        "测试集-宏平均F1-score": f"{macro_f1:.4f}",
        # 添加加权平均指标
        "测试集-加权Precision": f"{weighted_precision:.3f}",
        "测试集-加权Recall": f"{weighted_recall:.3f}",
        "测试集-加权F1-score": f"{weighted_f1:.3f}",
        "参数总量": f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    }

    # report = {
    #     "完成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "总训练轮数": args.epochs,
    #     "最佳验证准确率": f"{best_acc:.3f}%",
    #     "测试集准确率": f"{test_acc:.3f}%",
    #     "参数总量": f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    # }
    training_report = output_dir / "training_report.json"
    with open(training_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
