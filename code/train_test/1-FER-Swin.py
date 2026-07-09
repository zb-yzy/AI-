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
DATASET_NAME = "/root/autodl-tmp/dataset/FER-2013"
SAVE_NAME = "FER"
DATA_ROOT = f"{DATASET_NAME}"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results1/FER_Swin_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{SAVE_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{SAVE_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{SAVE_NAME}.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{SAVE_NAME}.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
# ======================================================

# ===================== Swin 窗口注意力基础模块 =====================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0, 3, 1, 2)
        return x

# ===================== 嵌入 Swin 注意力的 ResNet18 =====================
class ResNet18_Swin(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 复用原 ResNet 层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # ========== 核心修复：窗口大小=7，匹配7×7特征图，关闭移位 ==========
        self.swin_blocks = nn.Sequential(
            SwinTransformerBlock(dim=512, num_heads=16, window_size=7, shift_size=0),
            SwinTransformerBlock(dim=512, num_heads=16, window_size=7, shift_size=0)
        )

        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 插入窗口注意力
        x = self.swin_blocks(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(lab.cpu().numpy())
    return total_loss / total, correct / total, all_labels, all_preds

def write_log(epoch, train_loss, train_acc, val_loss, val_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch},train_loss={train_loss:.4f},train_acc={train_acc:.4f},val_loss={val_loss:.4f},val_acc={val_acc:.4f}\n")

# ===================== 保存测试集成功/失败图片 =====================
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

    print(f"✅ 测试集样例保存完成：成功 {success} 张，失败 {fail} 张")

# ===================== 绘制并保存混淆矩阵 =====================
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

    # ========== 使用 嵌入Swin窗口注意力的ResNet18 ==========
    model = ResNet18_Swin(num_classes=NUM_CLASSES)
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
    test_loss, test_acc, all_labels, all_preds = test_final(model, test_loader, criterion, DEVICE)
    print(f"📊 最终测试集结果：loss={test_loss:.4f}, acc={test_acc:.4f}")

    # 保存最终测试结果
    with open(os.path.join(SAVE_DIR, "final_test_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"测试集损失: {test_loss:.4f}\n测试集精度: {test_acc:.4f}")

    # ========== 绘制混淆矩阵 ==========
    print("\n📊 生成混淆矩阵...")
    class_names = test_dataset.classes
    plot_confusion_matrix(all_labels, all_preds, class_names, CM_SAVE_PATH)

    # ========== 保存测试集预测图片 ==========
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