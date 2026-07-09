import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===================== 【用户配置】 =====================
DATASET_NAME = "../../data/FER-2013"
SAVE_NAME = "FER"
DATA_ROOT = f"{DATASET_NAME}"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results1/FER_train_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{SAVE_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{SAVE_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{SAVE_NAME}.png")
# 新增：混淆矩阵保存路径
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{SAVE_NAME}.png")

# 新增：预测结果保存路径
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
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
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.2f}%")
    return total_loss / total, correct / total

def val(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="验证", leave=False)
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            loss = criterion(out, lab)
            total_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100*correct/total:.2f}%")
    return total_loss / total, correct / total

# 最终测试集专用（只在最后跑一次）
def test_final(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="最终测试", leave=False)
        for img, lab in pbar:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            loss = criterion(out, lab)
            total_loss += loss.item() * img.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)
            # 收集预测和标签，用于混淆矩阵
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
    return total_loss / total, correct / total, all_labels, all_preds

def write_log(epoch, train_loss, train_acc, val_loss, val_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch},train_loss={train_loss:.4f},train_acc={train_acc:.4f},val_loss={val_loss:.4f},val_acc={val_acc:.4f}\n")

# ===================== 【新增】保存测试集成功/失败图片 =====================
def save_test_examples(model, loader, device, save_dir, max_save=10):
    # 清空旧图片
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, f))
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    class_names = loader.dataset.classes
    success = 0
    fail = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            for i in range(imgs.size(0)):
                true_lab = labels[i].item()
                pred_lab = preds[i].item()
                img = imgs[i].cpu()

                # 反归一化，让图片正常显示
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                img = transforms.ToPILImage()(img)

                true_name = class_names[true_lab]
                pred_name = class_names[pred_lab]

                # 保存成功
                if true_lab == pred_lab and success < max_save:
                    path = os.path.join(save_dir, f"success_{success:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(path)
                    success += 1

                # 保存失败
                elif true_lab != pred_lab and fail < max_save:
                    path = os.path.join(save_dir, f"fail_{fail:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(path)
                    fail += 1

                if success >= max_save and fail >= max_save:
                    print(f"✅ 已保存：成功 {success} 张，失败 {fail} 张")
                    return

    print(f"✅ 测试集样例保存完成：成功 {success} 张，失败 {fail} 张")

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

# ===================== 主入口 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 加载数据集：train / val / test
    train_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), val_test_transform)

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")

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
    val_losses, val_accs = [], []

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
                    val_loss = float(parts[3].replace("val_loss=", ""))
                    val_acc = float(parts[4].replace("val_acc=", ""))
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    if val_acc > best_acc:
                        best_acc = val_acc
                except:
                    continue
        start_epoch = len(train_losses)
        print(f"✅ 读取历史完成 | 上次训练到 epoch {start_epoch} | 最佳精度 {best_acc:.4f}")

    # 加载模型
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ 已加载最优模型")

    # 开始训练
    print(f"\n🚀 训练开始 | 从 epoch {start_epoch+1} 训练 {CONTINUE_EPOCHS} 轮 | 设备：{DEVICE}\n")

    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = val(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        write_log(current_epoch, train_loss, train_acc, val_loss, val_acc)
        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | acc={best_acc:.4f}")

        print(f"训练 loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"验证 loss: {val_loss:.4f}  acc: {val_acc:.4f}\n")

    # ========== 训练全部结束后，在 test 上做最终评估 ==========
    print("\n========== 最终测试集评估 ==========")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    # 接收真实标签、预测标签
    test_loss, test_acc, all_labels, all_preds = test_final(model, test_loader, criterion, DEVICE)
    print(f"📊 最终测试集结果：loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 保存最终测试结果
    with open(os.path.join(SAVE_DIR, "final_test_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"测试集损失: {test_loss:.4f}\n测试集精度: {test_acc:.4f}")

    # ========== 【新增】绘制混淆矩阵 ==========
    print("\n📊 生成混淆矩阵...")
    class_names = test_dataset.classes
    plot_confusion_matrix(all_labels, all_preds, class_names, CM_SAVE_PATH)

    # ========== 【新增】保存测试集预测图片 ==========
    print("\n🖼️  开始保存测试集预测样例（成功/失败）...")
    save_test_examples(model, test_loader, DEVICE, PRED_SAVE_DIR, max_save=10)

    # 画图
    plt.figure(figsize=(12,5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="训练精度")
    plt.plot(val_accs, label="验证精度")
    plt.title("准确率曲线")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.show()

    print(f"\n🏁 训练完成！所有结果保存在: {SAVE_DIR}")
    print(f"📂 测试集样例保存在: {PRED_SAVE_DIR}")
    print(f"📂 混淆矩阵保存在: {CM_SAVE_PATH}")