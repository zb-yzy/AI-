import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===================== 【用户配置】 =====================
DATASET_NAME = "../../data/RAF"
RESULT_NAME  = "RAF"
DATA_ROOT = f"{DATASET_NAME}"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../../results/results1/RAF_train_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{RESULT_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{RESULT_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{RESULT_NAME}.png")
# 混淆矩阵保存路径
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{RESULT_NAME}.png")

# 新增：预测结果保存路径
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)

# ======================================================

# ===================== 训练 & 测试函数 =====================
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


def test(model, loader, criterion, device):
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
            # 收集预测与真实标签，用于混淆矩阵
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100 * correct / total:.2f}%")
    return total_loss / total, correct / total, all_labels, all_preds


def write_log(epoch, train_loss, train_acc, test_loss, test_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(
            f"Epoch {epoch},train_loss={train_loss:.4f},train_acc={train_acc:.4f},test_loss={test_loss:.4f},test_acc={test_acc:.4f}\n")


# ===================== 【新增】绘制并保存混淆矩阵 =====================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    # 按行归一化转为比例
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵(比例)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")


# ===================== 【新增】训练结束后保存预测样例 =====================
def save_test_examples(model, loader, device, save_dir, max_save=10):
    """
    只在训练最后执行一次：保存测试集预测成功和失败的图片
    :param max_save: 每种最多保存几张
    """
    # 先判断目录是否为空，避免报错
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))

    model.eval()
    class_names = loader.dataset.classes
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

                # 反归一化（方便看图）
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img = transforms.ToPILImage()(img_tensor)

                true_name = class_names[true_lab]
                pred_name = class_names[pred_lab]

                # 保存成功
                if true_lab == pred_lab and success_count < max_save:
                    save_path = os.path.join(save_dir,
                                             f"success_{success_count:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(save_path)
                    success_count += 1

                # 保存失败
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
    os.makedirs(PRED_SAVE_DIR, exist_ok=True)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "valid"), test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 构建模型
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # 恢复日志
    start_epoch = 0
    best_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("Epoch"):
                    continue

                parts = line.split(",")
                try:
                    epoch = int(parts[0].replace("Epoch ", ""))
                    train_loss = float(parts[1].replace("train_loss=", ""))
                    train_acc = float(parts[2].replace("train_acc=", ""))
                    test_loss = float(parts[3].replace("test_loss=", ""))
                    test_acc = float(parts[4].replace("test_acc=", ""))

                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    if test_acc > best_acc:
                        best_acc = test_acc
                except:
                    continue

        start_epoch = len(train_losses)
        print(f"✅ 读取历史完成 | 上次训练到 epoch {start_epoch} | 最佳精度 {best_acc:.4f}")

    # 加载模型
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ 已加载最优模型")

    # 开始训练
    print(
        f"\n🚀 训练开始 | 数据集: {DATASET_NAME} | 从 epoch {start_epoch + 1} 训练 {CONTINUE_EPOCHS} 轮|训练设备：{DEVICE}\n")

    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_loss, test_acc, _, _ = test(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        write_log(current_epoch, train_loss, train_acc, test_loss, test_acc)
        scheduler.step(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | acc={best_acc:.4f}")

        print(f"训练 loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"测试 loss: {test_loss:.4f}  acc: {test_acc:.4f}\n")

    # 画图：损失&精度曲线
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(test_losses, label="测试损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练精度")
    plt.plot(test_accs, label="测试精度")
    plt.title("准确率曲线")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.show()

    # ========== 生成混淆矩阵 ==========
    print("\n📊 开始生成混淆矩阵...")
    _, _, all_labels, all_preds = test(model, test_loader, criterion, DEVICE)
    class_names = test_dataset.classes
    plot_confusion_matrix(all_labels, all_preds, class_names, CM_SAVE_PATH)

    # ========== 保存预测样例 ==========
    print("\n📊 开始保存测试集预测样例（成功/失败）...")
    save_test_examples(model, test_loader, DEVICE, PRED_SAVE_DIR, max_save=10)

    print(f"\n🏁 训练完成！所有结果保存在: {SAVE_DIR}")
    print(f"📂 预测样例保存在: {PRED_SAVE_DIR}")
    print(f"📂 混淆矩阵保存在: {CM_SAVE_PATH}")