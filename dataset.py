# dataset.py
import os
import json
from torch.utils.data import Dataset
from PIL import Image

class ASLDataset(Dataset):
    def __init__(self, json_path, images_dir, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            coco = json.load(f)

        # 建立 category_id → 连续 label 0~25 的映射（跳过 id=0 的 "Letters"）
        self.cat_id_to_label = {}
        self.label_to_name = {}
        label_idx = 0
        for cat in coco['categories']:
            if cat['id'] == 0:  # 跳过父类
                continue
            self.cat_id_to_label[cat['id']] = label_idx
            self.label_to_name[label_idx] = cat['name']  # 0→'A', 1→'B', ...
            label_idx += 1

        # 建立 image_id → file_name 的快速查找
        self.img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

        # 构建样本列表 (img_path, label)
        self.samples = []
        for ann in coco['annotations']:
            img_id = ann['image_id']
            cat_id = ann['category_id']
            if cat_id not in self.cat_id_to_label:
                continue  # 理论上不会发生
            label = self.cat_id_to_label[cat_id]
            img_path = os.path.join(images_dir, self.img_id_to_filename[img_id])
            self.samples.append((img_path, label))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_label_name(self, label):
        return self.label_to_name.get(label, "Unknown")