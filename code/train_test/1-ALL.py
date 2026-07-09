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
DATASET_NAME = "../../data/all-two"
SAVE_NAME = "ALL"
DATA_ROOT = f"{DATASET_NAME}"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results1/All_train_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{SAVE_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{SAVE_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{SAVE_NAME}.png")
CM_SAVE_PATH_1 = os.path.join(SAVE_DIR, f"confusion_matrix_test1_{SAVE_NAME}.png")
CM_SAVE_PATH_2 = os.path.join(SAVE_DIR, f"confusion_matrix_test2_{SAVE_NAME}.png")
PRED_SAVE_DIR_1 = os.path.join(SAVE_DIR, "test1_predictions")
PRED_SAVE_DIR_2 = os.path.join(SAVE_DIR, "test2_predictions")
TEST_RESULT_PATH = os.path.join(SAVE_DIR, "final_test_result.txt")
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

def test_single(model, loader, criterion, device):
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
    return total_loss / total, correct / total, all_labels, all_preds

def write_log(epoch, train_loss, train_acc, test1_loss, test1_acc, test2_loss, test2_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch},train_loss={train_loss:.4f},train_acc={train_acc:.4f},"
                f"test1_loss={test1_loss:.4f},test1_acc={test1_acc:.4f},"
                f"test2_loss={test2_loss:.4f},test2_acc={test2_acc:.4f}\n")

# 保存测试样例
def save_test_examples(model, loader, device, save_dir, max_save=10):
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

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                img = transforms.ToPILImage()(img)

                true_name = class_names[true_lab]
                pred_name = class_names[pred_lab]

                if true_lab == pred_lab and success < max_save:
                    path = os.path.join(save_dir, f"success_{success:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(path)
                    success += 1
                elif true_lab != pred_lab and fail < max_save:
                    path = os.path.join(save_dir, f"fail_{fail:02d}_true_{true_name}_pred_{pred_name}.png")
                    img.save(path)
                    fail += 1

                if success >= max_save and fail >= max_save:
                    print(f"✅ 已保存：成功 {success} 张，失败 {fail} 张")
                    return
    print(f"✅ 测试样例保存完成：成功 {success} 张，失败 {fail} 张")

# 绘制归一化比例混淆矩阵
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

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # 1. 加载训练集 + 过滤隐藏目录
    train_dir = os.path.join(DATA_ROOT, "train")
    train_dataset = datasets.ImageFolder(
        train_dir, train_transform,
        is_valid_file=lambda x: not os.path.basename(x).startswith('.')
    )
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 2. 读取 test 下两个子文件夹，过滤隐藏目录
    test_root = os.path.join(DATA_ROOT, "test")
    test_sub_dirs = [
        d for d in os.listdir(test_root)
        if os.path.isdir(os.path.join(test_root, d)) and not d.startswith('.')
    ]
    test_dir1 = os.path.join(test_root, test_sub_dirs[0])
    test_dir2 = os.path.join(test_root, test_sub_dirs[1])

    test_dataset1 = datasets.ImageFolder(
        test_dir1, test_transform,
        is_valid_file=lambda x: not os.path.basename(x).startswith('.')
    )
    test_dataset2 = datasets.ImageFolder(
        test_dir2, test_transform,
        is_valid_file=lambda x: not os.path.basename(x).startswith('.')
    )
    test_loader1 = DataLoader(test_dataset1, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader2 = DataLoader(test_dataset2, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ 训练集: {len(train_dataset)}")
    print(f"✅ 测试集1({test_sub_dirs[0]}): {len(test_dataset1)}")
    print(f"✅ 测试集2({test_sub_dirs[1]}): {len(test_dataset2)}")

    # 构建模型
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 恢复日志与模型
    start_epoch = 0
    best_test_acc = 0.0
    train_losses, train_accs = [], []
    test1_losses, test1_accs = [], []
    test2_losses, test2_accs = [], []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Epoch"):
                    continue
                parts = line.split(",")
                try:
                    epoch = int(parts[0].replace("Epoch ", ""))
                    tr_loss = float(parts[1].replace("train_loss=", ""))
                    tr_acc = float(parts[2].replace("train_acc=", ""))
                    t1_loss = float(parts[3].replace("test1_loss=", ""))
                    t1_acc = float(parts[4].replace("test1_acc=", ""))
                    t2_loss = float(parts[5].replace("test2_loss=", ""))
                    t2_acc = float(parts[6].replace("test2_acc=", ""))

                    train_losses.append(tr_loss)
                    train_accs.append(tr_acc)
                    test1_losses.append(t1_loss)
                    test1_accs.append(t1_acc)
                    test2_losses.append(t2_loss)
                    test2_accs.append(t2_acc)

                    current_avg_test = (t1_acc + t2_acc) / 2
                    if current_avg_test > best_test_acc:
                        best_test_acc = current_avg_test
                except:
                    continue
        start_epoch = len(train_losses)
        print(f"✅ 读取历史完成 | 上次训练到 epoch {start_epoch}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ 已加载最优模型")

    # 开始训练
    print(f"\n🚀 训练开始 | 从 epoch {start_epoch+1} 训练 {CONTINUE_EPOCHS} 轮 | 设备：{DEVICE}\n")
    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        # 训练一轮
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # 每轮训练完，立刻在两个测试集上评估
        model.eval()
        t1_loss, t1_acc, _, _ = test_single(model, test_loader1, criterion, DEVICE)
        t2_loss, t2_acc, _, _ = test_single(model, test_loader2, criterion, DEVICE)
        model.train()

        # 保存曲线数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test1_losses.append(t1_loss)
        test1_accs.append(t1_acc)
        test2_losses.append(t2_loss)
        test2_accs.append(t2_acc)

        # 写入日志
        write_log(current_epoch, train_loss, train_acc, t1_loss, t1_acc, t2_loss, t2_acc)

        # 以两个测试集平均精度保存最优模型
        avg_test_acc = (t1_acc + t2_acc) / 2
        if avg_test_acc > best_test_acc:
            best_test_acc = avg_test_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | 平均测试精度={best_test_acc:.4f}")

        # 【关键】每轮实时打印三组指标
        print(f"训练集   loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"{test_sub_dirs[0]} loss: {t1_loss:.4f}  acc: {t1_acc:.4f}")
        print(f"{test_sub_dirs[1]} loss: {t2_loss:.4f}  acc: {t2_acc:.4f}\n")

    # ========== 最终整体评估 ==========
    print("\n========== 最终整体评估 ==========")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))

    train_final_loss, train_final_acc, _, _ = test_single(model, train_loader, criterion, DEVICE)
    loss1, acc1, labels1, preds1 = test_single(model, test_loader1, criterion, DEVICE)
    loss2, acc2, labels2, preds2 = test_single(model, test_loader2, criterion, DEVICE)

    print(f"📊 最终训练集准确率: {train_final_acc:.4f}")
    print(f"📊 测试集1({test_sub_dirs[0]}) 准确率: {acc1:.4f}")
    print(f"📊 测试集2({test_sub_dirs[1]}) 准确率: {acc2:.4f}")

    # 写入最终结果文件
    with open(TEST_RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(f"最终训练集:\n损失: {train_final_loss:.4f}\n精度: {train_final_acc:.4f}\n\n")
        f.write(f"测试集1({test_sub_dirs[0]}):\n损失: {loss1:.4f}\n精度: {acc1:.4f}\n\n")
        f.write(f"测试集2({test_sub_dirs[1]}):\n损失: {loss2:.4f}\n精度: {acc2:.4f}")

    # 混淆矩阵
    print("\n📊 生成混淆矩阵...")
    class_names = test_dataset1.classes
    plot_confusion_matrix(labels1, preds1, class_names, CM_SAVE_PATH_1)
    plot_confusion_matrix(labels2, preds2, class_names, CM_SAVE_PATH_2)

    # 保存预测样例
    print("\n🖼️  保存测试集1预测样例...")
    save_test_examples(model, test_loader1, DEVICE, PRED_SAVE_DIR_1, max_save=10)
    print("\n🖼️  保存测试集2预测样例...")
    save_test_examples(model, test_loader2, DEVICE, PRED_SAVE_DIR_2, max_save=10)

    # 绘制曲线（增加两条测试集曲线）
    plt.figure(figsize=(12,5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(test1_losses, label=f"{test_sub_dirs[0]}损失")
    plt.plot(test2_losses, label=f"{test_sub_dirs[1]}损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="训练精度")
    plt.plot(test1_accs, label=f"{test_sub_dirs[0]}精度")
    plt.plot(test2_accs, label=f"{test_sub_dirs[1]}精度")
    plt.title("准确率曲线")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.close()

    print(f"\n🏁 训练完成！所有结果保存在: {SAVE_DIR}")