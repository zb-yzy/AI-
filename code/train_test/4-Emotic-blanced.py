import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models.resnet import ResNet18_Weights
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from PIL import Image
import ast
from collections import defaultdict

# ===================== 用户配置 =====================
DATA_ROOT = "../../data/Emotic"
FACE_DIR = os.path.join(DATA_ROOT, "face")
CSV_DIR = os.path.join(DATA_ROOT, "csv")

SAVE_NAME = "Emotic_VAD_balanced"
SAVE_DIR = f"../../results/results4/{SAVE_NAME}"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{SAVE_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{SAVE_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{SAVE_NAME}.png")
EXAMPLES_SAVE_DIR = os.path.join(SAVE_DIR, "test_examples")

IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 均衡策略超参
BIN_NUM = 10          # 将VAD每个维度划分为10个区间统计分布
USE_WEIGHTED_LOSS = True  # 开启加权损失
USE_HUBER_LOSS = True      # 开启鲁棒Huber损失，降低密集区间主导
HUBER_DELTA = 0.5
USE_SAMPLER = True         # 开启加权采样，均衡每批次样本分布
# =====================================================

# ===================== CBAM 注意力模块 =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return attention * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def create_resnet18_with_cbam_regression(output_dim=3, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.cbam = CBAM(in_channels=in_features, reduction=16)
    model.fc = nn.Linear(in_features, output_dim)

    def new_forward(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.cbam(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        x = model.fc(x)
        return x

    model.forward = new_forward
    return model

# ===================== 加权损失函数（核心均衡模块） =====================
class WeightedHuberLoss(nn.Module):
    def __init__(self, delta=0.5):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target, sample_weight):
        diff = torch.abs(pred - target)
        # Huber损失
        loss = torch.where(diff < self.delta, 0.5 * diff ** 2, self.delta * (diff - 0.5 * self.delta))
        # 样本加权: 稀疏区间权重更大
        loss = loss * sample_weight.unsqueeze(1)
        return torch.mean(loss)

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, sample_weight):
        loss = (pred - target) ** 2
        loss = loss * sample_weight.unsqueeze(1)
        return torch.mean(loss)

# ===================== 增强数据集：输出标签 + 计算样本权重（已修复越界） =====================
class EmoticVADDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, bin_num=10, global_label_stats=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        self.labels_all = []
        self.bin_num = bin_num

        for idx, row in df.iterrows():
            img_name = row['Filename']
            img_path = self._find_image_path(img_name)
            if img_path is None:
                continue
            try:
                with Image.open(img_path) as test_img:
                    test_img.convert('RGB')
            except Exception:
                continue
            labels_str = row['Continuous_Labels']
            if isinstance(labels_str, str):
                labels_list = ast.literal_eval(labels_str)
            else:
                labels_list = labels_str
            labels = np.array(labels_list[:3], dtype=np.float32)
            self.samples.append((img_path, labels))
            self.labels_all.append(labels)

        self.labels_all = np.array(self.labels_all)
        print(f"✅ 有效样本数: {len(self.samples)} (原始CSV样本数: {len(df)})")

        # 计算每个样本的权重（基于标签分布）
        if global_label_stats is None:
            self.label_min = self.labels_all.min(axis=0)
            self.label_max = self.labels_all.max(axis=0)
        else:
            self.label_min, self.label_max = global_label_stats

        self.sample_weights = self._calc_sample_weights()

    def _calc_sample_weights(self):
        """根据VAD区间密度计算权重：样本越少的区间，权重越大（修复index越界）"""
        weights = np.ones(len(self.samples), dtype=np.float32)
        dim = 3
        max_bin_idx = self.bin_num - 1  # 最大合法下标

        for d in range(dim):
            # 划分区间
            bins = np.linspace(self.label_min[d], self.label_max[d], self.bin_num + 1)
            bin_cnt = np.histogram(self.labels_all[:, d], bins=bins)[0]
            # 区间密度倒数作为权重，避免除0
            bin_cnt[bin_cnt == 0] = 1
            bin_weight = 1.0 / bin_cnt
            # 匹配区间并截断下标，防止越界
            bin_idx = np.digitize(self.labels_all[:, d], bins) - 1
            bin_idx = np.clip(bin_idx, 0, max_bin_idx)
            weights *= bin_weight[bin_idx]

        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        return torch.from_numpy(weights).float()

    def get_weights(self):
        return self.sample_weights

    def get_label_range(self):
        return self.label_min, self.label_max

    def _find_image_path(self, filename):
        img_path = os.path.join(self.img_dir, filename)
        if os.path.exists(img_path):
            return img_path
        base = os.path.splitext(filename)[0]
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(self.img_dir, base + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, labels = self.samples[idx]
        label_tensor = torch.from_numpy(labels).float()
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_tensor, self.sample_weights[idx]

# ===================== 训练 & 评估函数（适配加权输入） =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="训练", leave=False)
    for img, lab, w in pbar:
        img, lab, w = img.to(device), lab.to(device), w.to(device)
        out = model(img)
        loss = criterion(out, lab, w)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="评估", leave=False)
        for img, lab, _ in pbar:
            img, lab = img.to(device), lab.to(device)
            out = model(img)
            loss = criterion(out, lab, torch.ones_like(lab[:,0]))
            total_loss += loss.item() * img.size(0)
            all_preds.append(out.cpu().numpy())
            all_labels.append(lab.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae_per_dim = mean_absolute_error(all_labels, all_preds, multioutput='raw_values')
    rmse_per_dim = np.sqrt(mean_squared_error(all_labels, all_preds, multioutput='raw_values'))
    return total_loss / len(loader.dataset), mae, rmse, mae_per_dim, rmse_per_dim, all_labels, all_preds

def write_log(epoch, train_loss, val_loss, val_mae, val_rmse):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch},train_loss={train_loss:.6f},val_loss={val_loss:.6f},val_MAE={val_mae:.6f},val_RMSE={val_rmse:.6f}\n")

# ===================== 保存回归样例 =====================
def save_regression_examples(model, loader, device, save_dir, max_save=50, horizontal=True):
    os.makedirs(save_dir, exist_ok=True)
    for f in os.listdir(save_dir):
        os.remove(os.path.join(save_dir, f))

    model.eval()
    all_info = []
    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader, desc="收集测试样本"):
            imgs = imgs.to(device)
            preds = model(imgs)
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            errors = np.mean(np.abs(preds_np - labels_np), axis=1)
            for i in range(imgs.size(0)):
                all_info.append((imgs[i].cpu(), labels_np[i], preds_np[i], errors[i]))

    all_info.sort(key=lambda x: x[3])
    half = max_save // 2
    good_samples = all_info[:half]
    bad_samples = all_info[-half:] if len(all_info) >= half else all_info[-len(all_info)//2:]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    def denorm(img_tensor):
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        return transforms.ToPILImage()(img)

    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    for idx, (img_tensor, true_lab, pred_lab, err) in enumerate(good_samples):
        img_pil = denorm(img_tensor)
        lines = [f"True VAD: ({true_lab[0]:.2f}, {true_lab[1]:.2f}, {true_lab[2]:.2f})",
                 f"Pred VAD: ({pred_lab[0]:.2f}, {pred_lab[1]:.2f}, {pred_lab[2]:.2f})",
                 f"MAE: {err:.4f}"]
        text_width = 0
        text_height = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            text_width = max(text_width, line_w)
            text_height += line_h + 5
        text_height += 10
        if horizontal:
            new_width = img_pil.width + text_width + 40
            new_height = max(img_pil.height, text_height + 20)
            new_img = Image.new('RGB', (new_width, new_height), color=(255,255,255))
            new_img.paste(img_pil, (0, 0))
            draw = ImageDraw.Draw(new_img)
            y_offset = 20
            for line in lines:
                draw.text((img_pil.width + 20, y_offset), line, fill=(0,0,0), font=font)
                y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + 5
        else:
            new_width = max(img_pil.width, text_width + 40)
            new_height = img_pil.height + text_height + 30
            new_img = Image.new('RGB', (new_width, new_height), color=(255,255,255))
            new_img.paste(img_pil, ((new_width - img_pil.width)//2, 0))
            draw = ImageDraw.Draw(new_img)
            y_offset = img_pil.height + 10
            for line in lines:
                draw.text((20, y_offset), line, fill=(0,0,0), font=font)
                y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + 5
        save_path = os.path.join(save_dir, f"good_{idx:02d}_err{err:.4f}.png")
        new_img.save(save_path)

    for idx, (img_tensor, true_lab, pred_lab, err) in enumerate(bad_samples):
        img_pil = denorm(img_tensor)
        lines = [f"True VAD: ({true_lab[0]:.2f}, {true_lab[1]:.2f}, {true_lab[2]:.2f})",
                 f"Pred VAD: ({pred_lab[0]:.2f}, {pred_lab[1]:.2f}, {pred_lab[2]:.2f})",
                 f"MAE: {err:.4f}"]
        text_width = 0
        text_height = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_w = bbox[2] - bbox[0]
            line_h = bbox[3] - bbox[1]
            text_width = max(text_width, line_w)
            text_height += line_h + 5
        text_height += 10
        if horizontal:
            new_width = img_pil.width + text_width + 40
            new_height = max(img_pil.height, text_height + 20)
            new_img = Image.new('RGB', (new_width, new_height), color=(255,255,255))
            new_img.paste(img_pil, (0, 0))
            draw = ImageDraw.Draw(new_img)
            y_offset = 20
            for line in lines:
                draw.text((img_pil.width + 20, y_offset), line, fill=(0,0,0), font=font)
                y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + 5
        else:
            new_width = max(img_pil.width, text_width + 40)
            new_height = img_pil.height + text_height + 30
            new_img = Image.new('RGB', (new_width, new_height), color=(255,255,255))
            new_img.paste(img_pil, ((new_width - img_pil.width)//2, 0))
            draw = ImageDraw.Draw(new_img)
            y_offset = img_pil.height + 10
            for line in lines:
                draw.text((20, y_offset), line, fill=(0,0,0), font=font)
                y_offset += font.getbbox(line)[3] - font.getbbox(line)[1] + 5
        save_path = os.path.join(save_dir, f"bad_{idx:02d}_err{err:.4f}.png")
        new_img.save(save_path)

    print(f"✅ 已保存 {len(good_samples)} 个好例子 + {len(bad_samples)} 个差例子 到 {save_dir}")

# ===================== 主入口 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1. 读取CSV
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"在 {CSV_DIR} 中没有找到CSV文件")
    df_list = []
    for csv_file in csv_files:
        file_path = os.path.join(CSV_DIR, csv_file)
        df = pd.read_csv(file_path)
        required_cols = ['Filename', 'Continuous_Labels']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"文件 {csv_file} 缺少列 {col}")
        df = df[['Filename', 'Continuous_Labels']].copy()
        df = df.dropna(subset=['Filename', 'Continuous_Labels'])
        df_list.append(df)
        print(f"加载 {csv_file}: {len(df)} 条")
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"总计加载 {len(full_df)} 条标注")

    # 2. 划分数据集
    train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    print(f"训练集原始样本数: {len(train_df)}")
    print(f"验证集原始样本数: {len(val_df)}")
    print(f"测试集原始样本数: {len(test_df)}")

    # 3. 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 全局标签范围（统一所有子集区间划分）
    all_labels_temp = []
    for _, row in full_df.iterrows():
        lab = ast.literal_eval(row['Continuous_Labels'])[:3]
        all_labels_temp.append(lab)
    all_labels_temp = np.array(all_labels_temp)
    global_min = all_labels_temp.min(axis=0)
    global_max = all_labels_temp.max(axis=0)
    print(f"全局VAD标签范围 min:{global_min}, max:{global_max}")

    # 构建数据集（带权重）
    train_dataset = EmoticVADDataset(train_df, FACE_DIR, transform=train_transform, bin_num=BIN_NUM, global_label_stats=(global_min, global_max))
    val_dataset = EmoticVADDataset(val_df, FACE_DIR, transform=val_test_transform, bin_num=BIN_NUM, global_label_stats=(global_min, global_max))
    test_dataset = EmoticVADDataset(test_df, FACE_DIR, transform=val_test_transform, bin_num=BIN_NUM, global_label_stats=(global_min, global_max))

    # 4. 构建DataLoader（加权采样）
    if USE_SAMPLER:
        train_sampler = WeightedRandomSampler(train_dataset.get_weights(), num_samples=len(train_dataset), replacement=True)
        train_loader = DataLoader(train_dataset, BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 5. 模型、损失函数、优化器
    model = create_resnet18_with_cbam_regression(output_dim=3, pretrained=True)
    model = model.to(DEVICE)

    # 选择损失函数
    if USE_WEIGHTED_LOSS and USE_HUBER_LOSS:
        criterion = WeightedHuberLoss(delta=HUBER_DELTA)
    elif USE_WEIGHTED_LOSS:
        criterion = WeightedMSELoss()
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 6. 恢复训练
    start_epoch = 0
    best_val_mae = float('inf')
    train_losses, val_losses, val_maes, val_rmses = [], [], [], []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.startswith("Epoch"):
                    continue
                parts = line.strip().split(',')
                try:
                    epoch = int(parts[0].replace("Epoch ", ""))
                    tr_loss = float(parts[1].replace("train_loss=", ""))
                    vl_loss = float(parts[2].replace("val_loss=", ""))
                    vl_mae = float(parts[3].replace("val_MAE=", ""))
                    vl_rmse = float(parts[4].replace("val_RMSE=", ""))
                    train_losses.append(tr_loss)
                    val_losses.append(vl_loss)
                    val_maes.append(vl_mae)
                    val_rmses.append(vl_rmse)
                    if vl_mae < best_val_mae:
                        best_val_mae = vl_mae
                except:
                    continue
        start_epoch = len(train_losses)
        print(f"✅ 读取历史完成 | 上次训练到 epoch {start_epoch}")

    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"✅ 已加载最优模型（验证MAE={best_val_mae:.6f})")

    # 7. 训练循环
    print(f"\n🚀 训练开始 | 从 epoch {start_epoch+1} 训练 {EPOCHS} 轮 | 设备：{DEVICE}\n")
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        current_epoch = epoch + 1
        print(f"======== Epoch {current_epoch} ========")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_mae, val_rmse, mae_per_dim, rmse_per_dim, _, _ = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        write_log(current_epoch, train_loss, val_loss, val_mae, val_rmse)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型 | 验证MAE={best_val_mae:.6f}")

        scheduler.step(val_loss)

        print(f"训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f} | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f}")
        print(f"各维度MAE (V, A, D): {mae_per_dim}")
        print(f"各维度RMSE (V, A, D): {rmse_per_dim}\n")

    # 8. 最终评估
    print("\n========== 最终评估（最佳模型） ==========")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

    val_loss, val_mae, val_rmse, val_mae_per_dim, val_rmse_per_dim, _, _ = evaluate(model, val_loader, criterion, DEVICE)
    print(f"验证集 MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
    print(f"验证集维度MAE (V,A,D): {val_mae_per_dim}")
    print(f"验证集维度RMSE (V,A,D): {val_rmse_per_dim}")

    test_loss, test_mae, test_rmse, test_mae_per_dim, test_rmse_per_dim, _, _ = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\n测试集 MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
    print(f"测试集维度MAE (V,A,D): {test_mae_per_dim}")
    print(f"测试集维度RMSE (V,A,D): {test_rmse_per_dim}")

    with open(os.path.join(SAVE_DIR, "final_results.txt"), 'w') as f:
        f.write("========== 验证集结果 ==========\n")
        f.write(f"MAE: {val_mae:.6f}\nRMSE: {val_rmse:.6f}\n维度MAE: {val_mae_per_dim}\n维度RMSE: {val_rmse_per_dim}\n\n")
        f.write("========== 测试集结果 ==========\n")
        f.write(f"MAE: {test_mae:.6f}\nRMSE: {test_rmse:.6f}\n维度MAE: {test_mae_per_dim}\n维度RMSE: {test_rmse_per_dim}\n")

    # 9. 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('损失曲线')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='验证MAE')
    plt.title('验证MAE曲线')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200)
    plt.close()

    # 10. 保存测试样例
    print("\n========== 保存测试集样例 ==========")
    save_regression_examples(model, test_loader, DEVICE, EXAMPLES_SAVE_DIR, max_save=50)
    print(f"\n🏁 全部完成！结果保存在: {SAVE_DIR}")