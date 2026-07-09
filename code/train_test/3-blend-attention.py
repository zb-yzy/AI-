import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from transformers import BertTokenizer, BertModel

# ===================== 全局随机种子（仅用于训练过程，数据集无随机） =====================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 【全局配置：改为新拆分后的固定数据集路径】 =====================
# 新扁平数据集路径（split_dataset.py输出目录）
NEW_DATA_ROOT = "../../data/movie_review"
TRAIN_IMG_DIR = os.path.join(NEW_DATA_ROOT, "train_imgs")
TEST_IMG_DIR = os.path.join(NEW_DATA_ROOT, "test_imgs")
TRAIN_META_CSV = os.path.join(NEW_DATA_ROOT, "train_meta.csv")
TEST_META_CSV = os.path.join(NEW_DATA_ROOT, "test_meta.csv")

# 任务参数（和原代码完全不变）
NUM_CLASSES = 7
IMG_SIZE = 224
MAX_SEQ_LEN = 128
BATCH_SIZE = 16
LR = 1e-4
CONTINUE_EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BERT 路径
BERT_MODEL_DIR = "../model"
# ResNet 预训练权重
PRETRAINED_RESNET_PATH = "../../results/results1/All_train_results/best_model_ALL.pth"

# 结果保存（同原代码）
SAVE_DIR = "../../results/results3/Multi_attention_fixed_dataset_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_multimodal_model.pth")
LOG_PATH = os.path.join(SAVE_DIR, "train_log.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, "train_curve.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, "confusion_matrix.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_SAVE_DIR, exist_ok=True)

# ======================================================

# ===================== 2. 多模态数据集类：图像 + 文本 + 标签（无改动） =====================
class MultimodalDataset(Dataset):
    def __init__(self, img_path_list, review_text_list, label_list, tokenizer, max_seq_len, img_transform):
        self.img_paths = img_path_list
        self.texts = review_text_list
        self.labels = label_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 图像
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.img_transform:
            image = self.img_transform(image)

        # 文本
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attn_mask = encoding["attention_mask"].squeeze(0)
        token_type_ids = encoding["token_type_ids"].squeeze(0)

        # 标签
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, input_ids, attn_mask, token_type_ids, label

# ===================== 3. 跨模态交叉注意力融合模型（完全复用原代码无修改） =====================
class CrossAttentionFusion(nn.Module):
    def __init__(self, num_classes, bert_dir, resnet_pretrain_path=None):
        super().__init__()
        # -------- 视觉分支 ResNet18 --------
        self.resnet = models.resnet18(weights=None)
        if resnet_pretrain_path and os.path.exists(resnet_pretrain_path):
            pretrain_dict = torch.load(resnet_pretrain_path, map_location=DEVICE, weights_only=True)
            del pretrain_dict["fc.weight"]
            del pretrain_dict["fc.bias"]
            self.resnet.load_state_dict(pretrain_dict, strict=False)
            print("✅ ResNet 迁移权重加载完成")

        # 冻结浅层，微调深层
        for name, param in self.resnet.named_parameters():
            if "conv1" in name or "bn1" in name or "layer1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.vis_raw_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # -------- 文本分支 BERT --------
        self.bert = BertModel.from_pretrained(bert_dir, local_files_only=True)
        self.text_raw_dim = self.bert.config.hidden_size

        # -------- 特征维度统一 --------
        self.hidden_dim = 256
        # 视觉特征投影
        self.vis_proj = nn.Sequential(
            nn.Linear(self.vis_raw_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # 文本特征投影
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_raw_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # -------- 多头交叉注意力（图像 ↔ 文本） --------
        self.num_heads = 8
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.2,
            batch_first=True
        )

        # -------- 融合 & 分类模块 --------
        self.fusion_drop = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.hidden_dim * 2, 512)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, image, input_ids, attention_mask, token_type_ids):
        # 1. 图像全局特征
        vis_feat = self.resnet(image)
        vis_feat = self.vis_proj(vis_feat)
        # 扩充维度适配注意力: [B, D] -> [B, 1, D]
        vis_seq = vis_feat.unsqueeze(1)

        # 2. 文本  token 序列特征
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_seq = bert_out.last_hidden_state  # [B, L, D]
        text_seq = self.text_proj(text_seq)

        # 3. 交叉注意力：图像 Query 关注文本 Key/Value
        attn_out, _ = self.attn(query=vis_seq, key=text_seq, value=text_seq)
        attn_feat = attn_out.squeeze(1)  # [B, D]

        # 4. 原始视觉特征 + 注意力增强特征 拼接融合
        fuse_feat = torch.cat([vis_feat, attn_feat], dim=1)

        # 5. 分类头
        x = self.fusion_drop(fuse_feat)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        logits = self.classifier(x)

        return logits

# ===================== 4. 【重写数据加载函数：无任何随机操作，直接读取拆分好的固定数据集】 =====================
def load_fixed_split_data():
    # 读取预拆分好的训练、测试元数据（数据集划分已离线固定，无随机）
    df_train_meta = pd.read_csv(TRAIN_META_CSV, encoding="utf-8-sig")
    df_test_meta = pd.read_csv(TEST_META_CSV, encoding="utf-8-sig")

    # 拼接完整图片路径
    train_img_paths = [os.path.join(TRAIN_IMG_DIR, name) for name in df_train_meta["new_img_name"]]
    test_img_paths = [os.path.join(TEST_IMG_DIR, name) for name in df_test_meta["new_img_name"]]

    train_texts = df_train_meta["review_text"].tolist()
    test_texts = df_test_meta["review_text"].tolist()

    train_raw_labels = df_train_meta["emotion_raw"].tolist()
    test_raw_labels = df_test_meta["emotion_raw"].tolist()

    # 标签编码（和原始编码映射完全一致）
    all_raw_labels = train_raw_labels + test_raw_labels
    le = LabelEncoder()
    le.fit(all_raw_labels)

    train_label_enc = le.transform(train_raw_labels)
    test_label_enc = le.transform(test_raw_labels)

    print(f"✅ 固定划分数据集加载完成 | 训练样本:{len(train_img_paths)} 测试样本:{len(test_img_paths)}")
    print(f"✅ 标签映射: {dict(zip(le.classes_, range(NUM_CLASSES)))}")

    # 图像Transform（和原代码完全一致）
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # BERT分词器
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR, local_files_only=True)

    # 构建Dataset
    train_ds = MultimodalDataset(train_img_paths, train_texts, train_label_enc, tokenizer, MAX_SEQ_LEN, train_transform)
    test_ds = MultimodalDataset(test_img_paths, test_texts, test_label_enc, tokenizer, MAX_SEQ_LEN, test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 类别权重（仅基于固定训练集统计，无随机）
    train_cls_cnt = Counter(train_label_enc)
    class_counts = np.zeros(NUM_CLASSES)
    for k, v in train_cls_cnt.items():
        class_counts[k] = v
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    return train_loader, test_loader, class_weights, le

# ===================== 5. 训练 & 测试函数（完全复用原代码无修改） =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="训练", leave=False)
    for img, input_ids, attn_mask, token_type_ids, label in pbar:
        img = img.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(img, input_ids, attn_mask, token_type_ids)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        _, pred = torch.max(logits, dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.2f}%")

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

def test_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="测试", leave=False)
        for img, input_ids, attn_mask, token_type_ids, label in pbar:
            img = img.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            logits = model(img, input_ids, attn_mask, token_type_ids)
            loss = criterion(logits, label)

            total_loss += loss.item() * img.size(0)
            _, pred = torch.max(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.2f}%")

    avg_loss = total_loss / total
    avg_acc = correct / total

    # 计算平衡准确率
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = np.array(all_labels) == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            c_correct = np.sum((np.array(all_preds) == c) & mask)
            per_class_acc.append(c_correct / mask.sum())
    balanced_acc = np.mean(per_class_acc)

    return avg_loss, avg_acc, balanced_acc, all_labels, all_preds

# ===================== 6. 日志、绘图、混淆矩阵（完全复用原代码无修改） =====================
def write_log(epoch, tr_loss, tr_acc, te_loss, te_acc, bal_acc):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch},train_loss={tr_loss:.4f},train_acc={tr_acc:.4f},test_loss={te_loss:.4f},test_acc={te_acc:.4f},balanced_acc={bal_acc:.4f}\n")

def plot_curve(train_loss, train_acc, test_loss, test_acc, bal_acc):
    plt.figure(figsize=(15, 5))
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.subplot(1, 3, 1)
    plt.plot(train_loss, label="训练损失")
    plt.plot(test_loss, label="测试损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_acc, label="训练准确率")
    plt.plot(test_acc, label="测试准确率")
    plt.title("总体准确率")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(bal_acc, label="类别平均准确率", color="green")
    plt.title("平衡准确率")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.close()

def plot_conf_mat(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵(归一化)")
    plt.tight_layout()
    plt.savefig(CM_SAVE_PATH, dpi=200)
    plt.close()
    print("✅ 混淆矩阵已保存")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 加载固定离线划分数据集（无任何随机操作）
    print("📊 加载离线预拆分固定多模态数据集，数据集层面无随机，可完全复现划分结果...")
    train_loader, test_loader, class_weights, label_encoder = load_fixed_split_data()
    print(f"训练集: {len(train_loader.dataset)} 样本 | 测试集: {len(test_loader.dataset)} 样本")

    # 2. 初始化交叉注意力融合模型（和原代码一致）
    print("\n🚀 初始化跨模态交叉注意力融合模型...")
    model = CrossAttentionFusion(NUM_CLASSES, BERT_MODEL_DIR, PRETRAINED_RESNET_PATH)
    model = model.to(DEVICE)

    # 3. 损失函数 + 分组差异化学习率（完全不变）
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 划分参数组：ResNet / BERT / 投影+注意力+融合分类头
    resnet_params = []
    bert_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "resnet" in name:
            resnet_params.append(param)
        elif "bert" in name:
            bert_params.append(param)
        else:
            fusion_params.append(param)

    # 分组学习率保持不变
    optimizer = optim.AdamW([
        {"params": resnet_params, "lr": 8e-5},
        {"params": bert_params, "lr": 2e-6},
        {"params": fusion_params, "lr": 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # 4. 续训初始化（逻辑不变）
    start_epoch = 0
    best_bal_acc = 0.0
    train_loss_list, train_acc_list = [], []
    test_loss_list, test_acc_list, bal_acc_list = [], [], []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Epoch"):
                    continue
                parts = line.split(",")
                tr_loss = float(parts[1].split("=")[1])
                tr_acc = float(parts[2].split("=")[1])
                te_loss = float(parts[3].split("=")[1])
                te_acc = float(parts[4].split("=")[1])
                bal_acc = float(parts[5].split("=")[1])
                train_loss_list.append(tr_loss)
                train_acc_list.append(tr_acc)
                test_loss_list.append(te_loss)
                test_acc_list.append(te_acc)
                bal_acc_list.append(bal_acc)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
        start_epoch = len(train_loss_list)
        print(f"✅ 读取历史日志，上次训练至 Epoch {start_epoch}，最佳平衡准确率: {best_bal_acc:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print("✅ 加载历史最优模型权重")

    # 5. 训练循环
    print(f"\n🚀 开始训练，共 {CONTINUE_EPOCHS} 轮")
    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"\n======== Epoch {current_epoch} ========")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, bal_acc, all_true, all_pred = test_one_epoch(model, test_loader, criterion, DEVICE)

        train_loss_list.append(tr_loss)
        train_acc_list.append(tr_acc)
        test_loss_list.append(te_loss)
        test_acc_list.append(te_acc)
        bal_acc_list.append(bal_acc)

        write_log(current_epoch, tr_loss, tr_acc, te_loss, te_acc, bal_acc)
        scheduler.step(bal_acc)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型保存，当前最佳平衡准确率: {best_bal_acc:.4f}")

        print(f"训练 Loss:{tr_loss:.4f} Acc:{tr_acc:.4f}")
        print(f"测试 Loss:{te_loss:.4f} Acc:{te_acc:.4f} Balanced_Acc:{bal_acc:.4f}")

    # 6. 绘制曲线 & 混淆矩阵
    print("\n📈 绘制训练曲线")
    plot_curve(train_loss_list, train_acc_list, test_loss_list, test_acc_list, bal_acc_list)

    print("\n📊 生成混淆矩阵")
    _, _, _, final_true, final_pred = test_one_epoch(model, test_loader, criterion, DEVICE)
    class_names = label_encoder.classes_
    plot_conf_mat(final_true, final_pred, class_names)

    # 7. 输出每类准确率
    per_cls_acc = []
    for c in range(NUM_CLASSES):
        mask = np.array(final_true) == c
        if mask.sum() == 0:
            per_cls_acc.append(np.nan)
        else:
            correct = np.sum((np.array(final_pred) == c) & mask)
            per_cls_acc.append(correct / mask.sum())
    print("\n📋 各类别准确率明细：")
    for idx, name in enumerate(class_names):
        print(f"{name:12s} : {per_cls_acc[idx]:.4f}")

    print(f"\n🏁 全部训练完成！最优平衡准确率: {best_bal_acc:.4f}")
    print(f"结果目录: {SAVE_DIR}")