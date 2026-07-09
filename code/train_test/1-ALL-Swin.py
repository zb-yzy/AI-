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

# ===================== 【全局配置】 =====================
DATASET_NAME = "../../data/all-two"
RESULT_NAME  = "ALL"
DATA_ROOT = f"{DATASET_NAME}"
NUM_CLASSES = 7
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
CONTINUE_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results1/All_train_swin_window"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{RESULT_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{RESULT_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{RESULT_NAME}.png")
CM_SAVE_PATH_1 = os.path.join(SAVE_DIR, f"confusion_matrix_test1_{RESULT_NAME}.png")
CM_SAVE_PATH_2 = os.path.join(SAVE_DIR, f"confusion_matrix_test2_{RESULT_NAME}.png")
PRED_SAVE_DIR_1 = os.path.join(SAVE_DIR, "test1_predictions")
PRED_SAVE_DIR_2 = os.path.join(SAVE_DIR, "test2_predictions")
TEST_RESULT_PATH = os.path.join(SAVE_DIR, "final_test_result.txt")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PRED_SAVE_DIR_1, exist_ok=True)
os.makedirs(PRED_SAVE_DIR_2, exist_ok=True)

# ===================== 【Swin 窗口注意力模块（无移位窗口）】 =====================
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

# ===================== 【嵌入无移位Swin窗口注意力的ResNet18】 =====================
class ResNet18_Swin(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 固定窗口7×7，无移位 shift_size=0，适配224输入下layer4输出7×7特征图
        self.swin_attn = nn.Sequential(
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

        x = self.swin_attn(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ===================== 训练 & 测试工具函数 =====================
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
    
def save_test_examples(model, loader, device, save_dir, max_save=10):
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

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.224]).view(3, 1, 1)
                img_tensor = img_tensor * std + mean
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img = transforms.ToPILImage()(img_tensor)

                true_name = class_names[true_lab]
                pred_name = class_names[pred_lab]

                if true_lab == pred_lab and success_count < max_save:
                    save_path = os.path.join(save_dir, f"success_{success_count:02d}_true_{true_name}_pred_{pred_name}.png")
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

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 数据增强与归一化
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.224])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.224])
    ])

    # 加载训练集 + 两个测试集
    train_dir = os.path.join(DATA_ROOT, "train")
    train_dataset = datasets.ImageFolder(
        train_dir, train_transform,
        is_valid_file=lambda x: not os.path.basename(x).startswith('.')
    )
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    test_root = os.path.join(DATA_ROOT, "test")
    test_sub_dirs = [d for d in os.listdir(test_root)
                     if os.path.isdir(os.path.join(test_root, d)) and not d.startswith('.')]
    test_dir1 = os.path.join(test_root, test_sub_dirs[0])
    test_dir2 = os.path.join(test_root, test_sub_dirs[1])

    test_dataset1 = datasets.ImageFolder(test_dir1, test_transform)
    test_dataset2 = datasets.ImageFolder(test_dir2, test_transform)
    test_loader1 = DataLoader(test_dataset1, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader2 = DataLoader(test_dataset2, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"✅ 训练集: {len(train_dataset)}")
    print(f"✅ 测试集1({test_sub_dirs[0]}): {len(test_dataset1)}")
    print(f"✅ 测试集2({test_sub_dirs[1]}): {len(test_dataset2)}")

    # 初始化模型：ResNet18 + 无移位固定窗口Swin注意力
    model = ResNet18_Swin(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # 断点续训初始化
    start_epoch = 0
    best_avg_acc = 0.0
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

                    current_avg = (t1_acc + t2_acc) / 2
                    if current_avg > best_avg_acc:
                        best_avg_acc = current_avg
                except:
                    continue
        start_epoch = len(train_losses)
        print(f"✅ 读取历史 | 上次训练至 epoch {start_epoch} | 历史最佳平均精度 {best_avg_acc:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
        print(f"✅ 加载历史最优模型")

    # 开始训练循环
    print(f"\n🚀 开始训练 | 输入尺寸 {IMG_SIZE}×{IMG_SIZE} | 固定7×7无移位窗口 | 共训练 {CONTINUE_EPOCHS} 轮\n")
    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        t1_loss, t1_acc, _, _ = test_single(model, test_loader1, criterion, DEVICE)
        t2_loss, t2_acc, _, _ = test_single(model, test_loader2, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test1_losses.append(t1_loss)
        test1_accs.append(t1_acc)
        test2_losses.append(t2_loss)
        test2_accs.append(t2_acc)

        write_log(current_epoch, train_loss, train_acc, t1_loss, t1_acc, t2_loss, t2_acc)
        scheduler.step((t1_acc + t2_acc) / 2)

        avg_acc = (t1_acc + t2_acc) / 2
        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | 两测试集平均精度: {best_avg_acc:.4f}")

        print(f"训练集   loss:{train_loss:.4f}  acc:{train_acc:.4f}")
        print(f"{test_sub_dirs[0]} loss:{t1_loss:.4f}  acc:{t1_acc:.4f}")
        print(f"{test_sub_dirs[1]} loss:{t2_loss:.4f}  acc:{t2_acc:.4f}\n")

    # 绘制损失&精度曲线
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="训练损失")
    plt.plot(test1_losses, label=test_sub_dirs[0]+"损失")
    plt.plot(test2_losses, label=test_sub_dirs[1]+"损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="训练精度")
    plt.plot(test1_accs, label=test_sub_dirs[0]+"精度")
    plt.plot(test2_accs, label=test_sub_dirs[1]+"精度")
    plt.title("准确率曲线")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.close()

    # 最终评估 + 混淆矩阵
    print("\n📊 最终模型评估 & 生成混淆矩阵")
    _, _, labels1, preds1 = test_single(model, test_loader1, criterion, DEVICE)
    _, _, labels2, preds2 = test_single(model, test_loader2, criterion, DEVICE)
    class_names = test_dataset1.classes

    plot_confusion_matrix(labels1, preds1, class_names, CM_SAVE_PATH_1)
    plot_confusion_matrix(labels2, preds2, class_names, CM_SAVE_PATH_2)

    # 保存预测样例
    print("\n🖼️  保存测试集预测样例")
    save_test_examples(model, test_loader1, DEVICE, PRED_SAVE_DIR_1, max_save=10)
    save_test_examples(model, test_loader2, DEVICE, PRED_SAVE_DIR_2, max_save=10)

    # 保存最终结果文本
    tr_final_loss, tr_final_acc, _, _ = test_single(model, train_loader, criterion, DEVICE)
    t1_final_loss, t1_final_acc, _, _ = test_single(model, test_loader1, criterion, DEVICE)
    t2_final_loss, t2_final_acc, _, _ = test_single(model, test_loader2, criterion, DEVICE)

    with open(TEST_RESULT_PATH, "w", encoding="utf-8") as f:
        f.write(f"训练集  损失:{tr_final_loss:.4f}  精度:{tr_final_acc:.4f}\n")
        f.write(f"{test_sub_dirs[0]} 损失:{t1_final_loss:.4f}  精度:{t1_final_acc:.4f}\n")
        f.write(f"{test_sub_dirs[1]} 损失:{t2_final_loss:.4f}  精度:{t2_final_acc:.4f}\n")

    print(f"\n🏁 全部训练流程结束，结果统一存放至：{SAVE_DIR}")