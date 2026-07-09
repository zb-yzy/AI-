import os
import re
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import shutil

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

# ===================== 全局随机种子（仅训练过程，数据集无随机） =====================
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 【全局配置：固定离线拆分数据集路径】 =====================
# 离线拆分后的扁平数据集（和之前split_dataset.py输出一致）
NEW_DATA_ROOT = "../../data/movie_review"
TRAIN_IMG_DIR = os.path.join(NEW_DATA_ROOT, "train_imgs")
TEST_IMG_DIR = os.path.join(NEW_DATA_ROOT, "test_imgs")
TRAIN_META_CSV = os.path.join(NEW_DATA_ROOT, "train_meta.csv")
TEST_META_CSV = os.path.join(NEW_DATA_ROOT, "test_meta.csv")

# 任务参数（与原Late Fusion代码完全对齐）
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

# 结果保存目录
SAVE_DIR = "../../results/results3/Multi_late_fixed_dataset_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_multimodal_model.pth")
LOG_PATH = os.path.join(SAVE_DIR, "train_log.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, "train_curve.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, "confusion_matrix.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
# 新增：分歧案例保存根目录
CASE_ROOT = os.path.join(SAVE_DIR, "case_analysis")
CORRECT_DISAGREE_DIR = os.path.join(CASE_ROOT, "correct_disagree")
WRONG_DISAGREE_DIR = os.path.join(CASE_ROOT, "wrong_disagree")

# 创建全部目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_SAVE_DIR, exist_ok=True)
os.makedirs(CASE_ROOT, exist_ok=True)
os.makedirs(CORRECT_DISAGREE_DIR, exist_ok=True)
os.makedirs(WRONG_DISAGREE_DIR, exist_ok=True)

# ======================================================

# ===================== 2. 多模态数据集类（无修改） =====================
class MultimodalDataset(Dataset):
    def __init__(self, img_path_list, review_text_list, label_list, img_name_list, tokenizer, max_seq_len, img_transform):
        self.img_paths = img_path_list
        self.texts = review_text_list
        self.labels = label_list
        self.img_names = img_name_list  # 保存原图文件名，用于导出案例
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

        # 标签、原图名称
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        raw_img_name = self.img_names[idx]

        return image, input_ids, attn_mask, token_type_ids, label, raw_img_name

# ===================== 3. 晚期融合模型 Late Fusion（原代码完整保留） =====================
class LateFusionModel(nn.Module):
    def __init__(self, num_classes, bert_dir, resnet_pretrain_path=None):
        super().__init__()
        # -------- 视觉分支 ResNet18（独立分类头） --------
        self.resnet = models.resnet18(weights=None)
        if resnet_pretrain_path and os.path.exists(resnet_pretrain_path):
            pretrain_dict = torch.load(resnet_pretrain_path, map_location=DEVICE, weights_only=True)
            del pretrain_dict["fc.weight"]
            del pretrain_dict["fc.bias"]
            self.resnet.load_state_dict(pretrain_dict, strict=False)
            print("✅ ResNet 迁移权重加载完成")

        # 冻结策略：仅冻结浅层
        for name, param in self.resnet.named_parameters():
            if "conv1" in name or "bn1" in name or "layer1" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # ResNet 独立分类头
        vis_feat_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(vis_feat_dim, num_classes)

        # -------- 文本分支 BERT（独立分类头） --------
        self.bert = BertModel.from_pretrained(bert_dir, local_files_only=True)
        text_hidden_dim = self.bert.config.hidden_size
        self.text_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(text_hidden_dim, num_classes)
        )

        # -------- 可学习融合权重 --------
        self.alpha = nn.Parameter(torch.tensor([0.5]))  # 图像分支权重
        self.beta = nn.Parameter(torch.tensor([0.5]))   # 文本分支权重

    def forward(self, image, input_ids, attention_mask, token_type_ids):
        # 图像分支预测 logits
        vis_logits = self.resnet(image)

        # 文本分支预测 logits
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_cls_feat = bert_out.last_hidden_state[:, 0, :]
        text_logits = self.text_classifier(text_cls_feat)

        # 晚期融合：加权融合两路输出
        fuse_logits = self.alpha * vis_logits + self.beta * text_logits
        return fuse_logits, vis_logits, text_logits

# ===================== 4. 加载固定离线数据集（无随机，读取meta csv） =====================
def load_fixed_split_data():
    df_train_meta = pd.read_csv(TRAIN_META_CSV, encoding="utf-8-sig")
    df_test_meta = pd.read_csv(TEST_META_CSV, encoding="utf-8-sig")

    # 训练集
    train_img_names = df_train_meta["new_img_name"].tolist()
    train_img_paths = [os.path.join(TRAIN_IMG_DIR, name) for name in train_img_names]
    train_texts = df_train_meta["review_text"].tolist()
    train_raw_labels = df_train_meta["emotion_raw"].tolist()

    # 测试集
    test_img_names = df_test_meta["new_img_name"].tolist()
    test_img_paths = [os.path.join(TEST_IMG_DIR, name) for name in test_img_names]
    test_texts = df_test_meta["review_text"].tolist()
    test_raw_labels = df_test_meta["emotion_raw"].tolist()
    test_movie_ids = df_test_meta["movie_id"].tolist()  # 保存电影ID用于案例说明

    # 标签编码
    all_raw_labels = train_raw_labels + test_raw_labels
    le = LabelEncoder()
    le.fit(all_raw_labels)
    train_label_enc = le.transform(train_raw_labels)
    test_label_enc = le.transform(test_raw_labels)

    print(f"✅ 固定数据集加载完成 | 训练:{len(train_img_paths)} 测试:{len(test_img_paths)}")
    print(f"✅ 标签映射: {dict(zip(le.classes_, range(NUM_CLASSES)))}")

    # 图像增强
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

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
    train_ds = MultimodalDataset(train_img_paths, train_texts, train_label_enc, train_img_names, tokenizer, MAX_SEQ_LEN, train_transform)
    test_ds = MultimodalDataset(test_img_paths, test_texts, test_label_enc, test_img_names, tokenizer, MAX_SEQ_LEN, test_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 类别权重
    train_cls_cnt = Counter(train_label_enc)
    class_counts = np.zeros(NUM_CLASSES)
    for k, v in train_cls_cnt.items():
        class_counts[k] = v
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # 额外返回测试集元数据，导出案例使用
    test_meta_info = {
        "img_paths": test_img_paths,
        "img_names": test_img_names,
        "texts": test_texts,
        "movie_ids": test_movie_ids,
        "raw_labels": test_raw_labels
    }
    return train_loader, test_loader, class_weights, le, test_meta_info

# ===================== 5. 训练、测试函数 =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="训练", leave=False)
    for img, input_ids, attn_mask, token_type_ids, label, _ in pbar:
        img = img.to(device)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        fuse_logits, _, _ = model(img, input_ids, attn_mask, token_type_ids)
        loss = criterion(fuse_logits, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        _, pred = torch.max(fuse_logits, dim=1)
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
    all_fuse_preds = []
    all_vis_preds = []
    all_text_preds = []
    all_labels = []
    all_img_names = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="测试", leave=False)
        for img, input_ids, attn_mask, token_type_ids, label, img_name in pbar:
            img = img.to(device)
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)

            fuse_logits, vis_logits, text_logits = model(img, input_ids, attn_mask, token_type_ids)
            loss = criterion(fuse_logits, label)

            total_loss += loss.item() * img.size(0)
            _, fuse_pred = torch.max(fuse_logits, dim=1)
            _, vis_pred = torch.max(vis_logits, dim=1)
            _, text_pred = torch.max(text_logits, dim=1)

            correct += (fuse_pred == label).sum().item()
            total += label.size(0)

            all_fuse_preds.extend(fuse_pred.cpu().numpy())
            all_vis_preds.extend(vis_pred.cpu().numpy())
            all_text_preds.extend(text_pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_img_names.extend(img_name)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.2f}%")

    avg_loss = total_loss / total
    avg_acc = correct / total

    # 平衡准确率
    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = np.array(all_labels) == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            c_correct = np.sum((np.array(all_fuse_preds) == c) & mask)
            per_class_acc.append(c_correct / mask.sum())
    balanced_acc = np.mean(per_class_acc)

    return avg_loss, avg_acc, balanced_acc, all_labels, all_fuse_preds, all_vis_preds, all_text_preds, all_img_names

# ===================== 6. 日志、绘图、混淆矩阵函数 =====================
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

# ===================== 7. 新增：导出图像/文本预测分歧案例 =====================
def export_disagree_cases(
    all_true, all_fuse, all_vis, all_text, all_img_names,
    test_meta_info, label_encoder, case_num=30
):
    class_map = dict(zip(range(NUM_CLASSES), label_encoder.classes_))
    img_paths = test_meta_info["img_paths"]
    texts = test_meta_info["texts"]
    movie_ids = test_meta_info["movie_ids"]
    raw_labels = test_meta_info["raw_labels"]

    correct_disagree_list = []
    wrong_disagree_list = []

    # 遍历全部测试样本，筛选图像、文本预测不一致的样本
    for idx in range(len(all_true)):
        y_true = all_true[idx]
        y_fuse = all_fuse[idx]
        y_vis = all_vis[idx]
        y_text = all_text[idx]
        img_name = all_img_names[idx]

        # 图像分支预测 != 文本分支预测
        if y_vis != y_text:
            info = {
                "idx": idx,
                "img_path": img_paths[idx],
                "img_name": img_name,
                "movie_id": movie_ids[idx],
                "text": texts[idx],
                "true_label_id": y_true,
                "true_label": raw_labels[idx],
                "vis_pred_id": y_vis,
                "vis_pred": class_map[y_vis],
                "text_pred_id": y_text,
                "text_pred": class_map[y_text],
                "fuse_pred_id": y_fuse,
                "fuse_pred": class_map[y_fuse]
            }
            if y_fuse == y_true:
                correct_disagree_list.append(info)
            else:
                wrong_disagree_list.append(info)

    print(f"\n📊 分歧样本统计：")
    print(f"图像文本预测不一致且融合预测正确：{len(correct_disagree_list)} 例")
    print(f"图像文本预测不一致且融合预测错误：{len(wrong_disagree_list)} 例")

    # 随机选取指定数量案例
    random.shuffle(correct_disagree_list)
    random.shuffle(wrong_disagree_list)
    select_correct = correct_disagree_list[:case_num]
    select_wrong = wrong_disagree_list[:case_num]

    # 导出正确分歧案例
    for case_idx, case_info in enumerate(select_correct):
        case_folder = os.path.join(CORRECT_DISAGREE_DIR, f"case_{case_idx:03d}")
        os.makedirs(case_folder, exist_ok=True)
        # 复制原图
        src_img = case_info["img_path"]
        dst_img = os.path.join(case_folder, case_info["img_name"])
        shutil.copy(src_img, dst_img)
        # 写入说明文本
        txt_path = os.path.join(case_folder, "info.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"案例序号：{case_idx:03d}\n")
            f.write(f"电影ID：{case_info['movie_id']}\n")
            f.write(f"原图文件名：{case_info['img_name']}\n")
            f.write(f"真实情绪标签：{case_info['true_label']} (id:{case_info['true_label_id']})\n")
            f.write(f"图像分支单独预测：{case_info['vis_pred']} (id:{case_info['vis_pred_id']})\n")
            f.write(f"文本分支单独预测：{case_info['text_pred']} (id:{case_info['text_pred_id']})\n")
            f.write(f"晚期融合最终预测：{case_info['fuse_pred']} (id:{case_info['fuse_pred_id']})\n")
            f.write("判定：融合预测正确（图像、文本预测出现分歧但融合后纠正）\n")
            f.write(f"\n评论原文：\n{case_info['text']}\n")

    # 导出错误分歧案例
    for case_idx, case_info in enumerate(select_wrong):
        case_folder = os.path.join(WRONG_DISAGREE_DIR, f"case_{case_idx:03d}")
        os.makedirs(case_folder, exist_ok=True)
        src_img = case_info["img_path"]
        dst_img = os.path.join(case_folder, case_info["img_name"])
        shutil.copy(src_img, dst_img)
        txt_path = os.path.join(case_folder, "info.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"案例序号：{case_idx:03d}\n")
            f.write(f"电影ID：{case_info['movie_id']}\n")
            f.write(f"原图文件名：{case_info['img_name']}\n")
            f.write(f"真实情绪标签：{case_info['true_label']} (id:{case_info['true_label_id']})\n")
            f.write(f"图像分支单独预测：{case_info['vis_pred']} (id:{case_info['vis_pred_id']})\n")
            f.write(f"文本分支单独预测：{case_info['text_pred']} (id:{case_info['text_pred_id']})\n")
            f.write(f"晚期融合最终预测：{case_info['fuse_pred']} (id:{case_info['fuse_pred_id']})\n")
            f.write("判定：融合预测错误（图像、文本预测分歧，融合后依然预测失败）\n")
            f.write(f"\n评论原文：\n{case_info['text']}\n")

    print(f"✅ 分歧案例导出完成！")
    print(f"正确分歧案例（30例）路径：{CORRECT_DISAGREE_DIR}")
    print(f"错误分歧案例（30例）路径：{WRONG_DISAGREE_DIR}")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 1. 加载固定离线数据集
    print("📊 加载离线预拆分固定多模态数据集...")
    train_loader, test_loader, class_weights, label_encoder, test_meta_info = load_fixed_split_data()
    print(f"训练集: {len(train_loader.dataset)} 样本 | 测试集: {len(test_loader.dataset)} 样本")

    # 2. 初始化晚期融合模型
    print("\n🚀 初始化晚期融合多模态模型...")
    model = LateFusionModel(NUM_CLASSES, BERT_MODEL_DIR, PRETRAINED_RESNET_PATH)
    model = model.to(DEVICE)

    # 3. 损失函数 + 分组差异化学习率（适配Late Fusion参数）
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            # text_classifier、alpha、beta归融合分支
            fusion_params.append(param)

    optimizer = optim.AdamW([
        {"params": resnet_params, "lr": 8e-5},
        {"params": bert_params, "lr": 2e-6},
        {"params": fusion_params, "lr": 1e-4}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # 4. 续训加载
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
        te_loss, te_acc, bal_acc, all_true, all_fuse, all_vis, all_text, all_img_names = test_one_epoch(model, test_loader, criterion, DEVICE)

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

    # 6. 绘图、混淆矩阵
    print("\n📈 绘制训练曲线")
    plot_curve(train_loss_list, train_acc_list, test_loss_list, test_acc_list, bal_acc)

    print("\n📊 生成混淆矩阵")
    _, _, _, final_true, final_fuse, final_vis, final_text, final_img_names = test_one_epoch(model, test_loader, criterion, DEVICE)
    class_names = label_encoder.classes_
    plot_conf_mat(final_true, final_fuse, class_names)

    # 7. 输出每类准确率
    per_cls_acc = []
    for c in range(NUM_CLASSES):
        mask = np.array(final_true) == c
        if mask.sum() == 0:
            per_cls_acc.append(np.nan)
        else:
            correct = np.sum((np.array(final_fuse) == c) & mask)
            per_cls_acc.append(correct / mask.sum())
    print("\n📋 各类别准确率明细：")
    for idx, name in enumerate(class_names):
        print(f"{name:12s} : {per_cls_acc[idx]:.4f}")

    # 8. 导出图像文本预测分歧案例（核心新增功能）
    print("\n📂 开始导出图像/文本分支预测不一致案例...")
    export_disagree_cases(
        all_true=final_true,
        all_fuse=final_fuse,
        all_vis=final_vis,
        all_text=final_text,
        all_img_names=final_img_names,
        test_meta_info=test_meta_info,
        label_encoder=label_encoder,
        case_num=30
    )

    print(f"\n🏁 全部训练与分析完成！最优平衡准确率: {best_bal_acc:.4f}")
    print(f"结果总目录: {SAVE_DIR}")