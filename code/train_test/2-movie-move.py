import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===================== 【用户配置 - 无任何随机，与原代码超参完全对齐】 =====================
# 平铺数据集路径（之前split_dataset生成的）
FLAT_DATA_ROOT = "../../data/movie"
TRAIN_DIR = os.path.join(FLAT_DATA_ROOT, "train")
TEST_DIR = os.path.join(FLAT_DATA_ROOT, "test")

RESULT_NAME = "Movie_Flat_NoRand_Move"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 迁移学习：源模型预训练权重路径
PRETRAINED_MODEL_PATH = "../../results/results1/All_train_results/best_model_ALL.pth"

SAVE_DIR = "../../results/results2/Movie_move_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{RESULT_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{RESULT_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{RESULT_NAME}.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{RESULT_NAME}.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
os.makedirs(SAVE_DIR, exist_ok=True)


# ==================================================================================

# ===================== 自定义平铺数据集：从文件名解析标签 =====================
class FlatImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 固定排序，消除加载随机
        self.file_list = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(("jpg", "jpeg", "png"))
        ])
        # 提取全部标签并固定排序，与原ImageFolder类别顺序一致
        label_set = set()
        for fname in self.file_list:
            tag = fname.split("_")[-1].split(".")[0]
            label_set.add(tag)
        self.class_names = sorted(list(label_set))
        self.label2idx = {cls: i for i, cls in enumerate(self.class_names)}
        # 预存全部标签，用于计算类别权重
        self.all_labels = []
        for fname in self.file_list:
            tag = fname.split("_")[-1].split(".")[0]
            self.all_labels.append(self.label2idx[tag])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        img_path = os.path.join(self.data_dir, fname)
        img = Image.open(img_path).convert("RGB")
        label_tag = fname.split("_")[-1].split(".")[0]
        label = self.label2idx[label_tag]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ===================== 训练 & 测试函数（与原版完全一致） =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="训练", leave=False)
    for img, lab in pbar:
        img, lab = img.to(device), lab.to(device)
        out = model(img)
        loss = criterion(out, lab)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == lab).sum().item()
        total += lab.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100 * correct / total:.2f}%")
    return total_loss / total, correct / total


def test(model, loader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="测试", leave=False)
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            loss = criterion(out, lab)
            total_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100 * correct / total:.2f}%")

    overall_acc = correct / total
    per_class_acc = []
    for c in range(num_classes):
        mask = np.array(all_labels) == c
        if mask.sum() == 0:
            per_class_acc.append(0.0)
        else:
            class_correct = np.sum((np.array(all_preds) == c) & mask)
            per_class_acc.append(class_correct / mask.sum())
    balanced_acc = np.mean(per_class_acc)
    return total_loss / total, overall_acc, balanced_acc, all_labels, all_preds


def write_log(epoch, train_loss, train_acc, test_loss, test_acc, balanced_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(
            f"Epoch {epoch},train_loss={train_loss:.4f},train_acc={train_acc:.4f},"
            f"test_loss={test_loss:.4f},test_acc={test_acc:.4f},balanced_acc={balanced_acc:.4f}\n"
        )


# ===================== 绘制并保存混淆矩阵 =====================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵(比例)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")


def save_test_examples(model, loader, device, save_dir, max_save=10):
    # 修复：先创建文件夹，保证目录存在
    os.makedirs(save_dir, exist_ok=True)
    # 再清空内部图片，不删除文件夹本身
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            file_path = os.path.join(save_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

    model.eval()
    class_names = loader.dataset.class_names
    success_count = 0
    fail_count = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for i in range(imgs.size(0)):
                true_lab = labels[i].item()
                pred_lab = preds[i].item()
                img_tensor = imgs[i].cpu()

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img = transforms.ToPILImage()(img_tensor)

                true_name = class_names[true_lab]
                pred_name = class_names[pred_lab]

                if true_lab == pred_lab and success_count < max_save:
                    save_path = os.path.join(save_dir,
                                             f"success_{success_count:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(save_path)
                    success_count += 1
                elif true_lab != pred_lab and fail_count < max_save:
                    save_path = os.path.join(save_dir, f"fail_{fail_count:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(save_path)
                    fail_count += 1

                if success_count >= max_save and fail_count >= max_save:
                    print(f"✅ 已保存成功图片 {success_count} 张，失败图片 {fail_count} 张")
                    return

    print(f"✅ 预测样例保存完成：成功 {success_count} 张，失败 {fail_count} 张")


# ===================== 主入口 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---------- 无随机数据增强：移除随机翻转、随机旋转 ----------
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载平铺数据集
    train_dataset = FlatImageDataset(TRAIN_DIR, transform=train_transform)
    test_dataset = FlatImageDataset(TEST_DIR, transform=test_transform)
    class_names = train_dataset.class_names

    print(f"📊 平铺数据集加载完成")
    print(f"类别列表: {class_names}")
    print(f"训练集样本数: {len(train_dataset)} | 测试集样本数: {len(test_dataset)}")

    # shuffle=False 完全固定读取顺序，无任何随机
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- 计算类别权重（和原代码逻辑一致） ----------
    train_labels = np.array(train_dataset.all_labels)
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"📊 类别权重: {class_weights.cpu().numpy()}")

    # ===================== 迁移学习权重加载（与原版完全一致） =====================
    model = models.resnet18(weights=None)

    if os.path.exists(PRETRAINED_MODEL_PATH):
        pretrain_state = torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True)
        del pretrain_state["fc.weight"]
        del pretrain_state["fc.bias"]
        model.load_state_dict(pretrain_state, strict=False)
        print(f"✅ 成功加载源域预训练主干权重: {PRETRAINED_MODEL_PATH}")
    else:
        print(f"⚠️ 未找到源域预训练权重 {PRETRAINED_MODEL_PATH}，模型随机初始化")

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # 冻结策略：冻结 conv1 + bn1 + layer1~layer3，仅训练 layer4 + fc
    for name, param in model.named_parameters():
        if "conv1" in name or "bn1" in name or "layer1" in name or "layer2" in name or "layer3" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"总参数量: {total_params:,} | 可训练参数量: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")

    model = model.to(DEVICE)
    # ==========================================================================

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # ---------- 恢复训练日志与模型 ----------
    start_epoch = 0
    best_balanced_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    balanced_accs = []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Epoch"):
                    continue
                parts = line.split(",")
                try:
                    epoch = int(parts[0].replace("Epoch ", ""))
                    train_loss = float(parts[1].replace("train_loss=", ""))
                    train_acc = float(parts[2].replace("train_acc=", ""))
                    test_loss = float(parts[3].replace("test_loss=", ""))
                    test_acc = float(parts[4].replace("test_acc=", ""))
                    if len(parts) >= 6:
                        balanced_acc = float(parts[5].replace("balanced_acc=", ""))
                    else:
                        balanced_acc = test_acc
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    balanced_accs.append(balanced_acc)
                    if balanced_acc > best_balanced_acc:
                        best_balanced_acc = balanced_acc
                except:
                    continue
        start_epoch = len(train_losses)
        print(f"✅ 读取历史完成 | 上次训练到 epoch {start_epoch} | 最佳平均准确率 {best_balanced_acc:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ 已加载最优模型")

    # ---------- 训练循环 ----------
    print(f"\n🚀 训练开始 | 平铺无随机数据集 | 从 epoch {start_epoch + 1} 训练 {CONTINUE_EPOCHS} 轮 | 设备：{DEVICE}\n")

    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, balanced_acc, _, _ = test(model, test_loader, criterion, DEVICE, NUM_CLASSES)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        balanced_accs.append(balanced_acc)

        write_log(current_epoch, train_loss, train_acc, test_loss, test_acc, balanced_acc)
        scheduler.step(balanced_acc)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | 平均准确率={best_balanced_acc:.4f}")

        print(f"训练 loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"测试 loss: {test_loss:.4f} 总体准确率: {test_acc:.4f} 平均准确率: {balanced_acc:.4f}\n")

    # ---------- 绘制三曲线图 ----------
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(test_losses, label="测试损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label="训练总体准确率")
    plt.plot(test_accs, label="测试总体准确率")
    plt.title("总体准确率曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(balanced_accs, label="测试平均准确率 (每个类别准确率均值)", color='green')
    plt.title("平均准确率曲线")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.close()

    # ---------- 混淆矩阵 & 预测样例 & 各类精度打印 ----------
    print("\n📊 开始生成混淆矩阵...")
    _, _, _, all_labels, all_preds = test(model, test_loader, criterion, DEVICE, NUM_CLASSES)
    plot_confusion_matrix(all_labels, all_preds, class_names, CM_SAVE_PATH)

    print("\n📊 开始保存测试集预测样例（成功/失败）...")
    save_test_examples(model, test_loader, DEVICE, PRED_SAVE_DIR, max_save=10)

    per_class_acc = []
    for c in range(NUM_CLASSES):
        mask = np.array(all_labels) == c
        if mask.sum() == 0:
            per_class_acc.append(float('nan'))
        else:
            class_correct = np.sum((np.array(all_preds) == c) & mask)
            per_class_acc.append(class_correct / mask.sum())
    print("\n📈 各类别准确率明细：")
    for i, name in enumerate(class_names):
        if np.isnan(per_class_acc[i]):
            print(f"  {name:10s} : 无测试样本")
        else:
            print(f"  {name:10s} : {per_class_acc[i]:.4f}")

    print(f"\n🏁 训练完成！所有结果保存在: {SAVE_DIR}")
    print(f"📂 预测样例保存在: {PRED_SAVE_DIR}")
    print(f"📂 混淆矩阵保存在: {CM_SAVE_PATH}")