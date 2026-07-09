import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from PIL import ImageFont, ImageDraw, Image
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mtcnn import MTCNN

# ===================== 配置（已按图片文件夹顺序修正标签） =====================
CLASS_MODEL_PATH = "../../results-model/model1/best_model_ALL.pth"
VAD_MODEL_PATH = "../../results-model/model2/best_model_Emotic_VAD_balanced.pth"
VIDEO_INPUT_PATH = "../../data/vidio/test_video.mp4"
VIDEO_OUTPUT_PATH = "../../results/vidio/self-regulate/output_analyzed.mp4"
JUMP_CURVE_SAVE_PATH = "../../results/vidio/self-regulate/final_jump_curve.png"

NUM_CLASSES = 7
AVG_NEIGHBOR_RANGE = 11
# 推理时序平滑窗口
SMOOTH_WINDOW = 15
# 时序损失权重，越大约束越平滑
TEMPORAL_LOSS_WEIGHT = 0.4

# 严格对应图片从上到下文件夹顺序：anger、disgust、fear、happy、normal、sad、surprised
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_NAMES_CN = ["愤怒", "厌恶", "惧怕", "开心", "平淡", "悲伤", "惊喜"]

# VAD区间同步对应修正后的7类情绪
VAD_RANGES_01 = [
    [0.7, 1.0, 0.15, 0.4, 0.7, 1.0],    # anger 愤怒
    [0.3, 0.7, 0.2, 0.45, 0.2, 0.6],    # disgust 厌恶
    [0.5, 0.9, 0.1, 0.4, 0.2, 0.5],     # fear 惧怕
    [0.3, 0.9, 0.8, 1.0, 0.6, 0.85],    # happy 开心
    [0.0, 0.2, 0.4, 0.6, 0.0, 0.4],     # normal 平淡
    [0.4, 0.85, 0.0, 0.2, 0.0, 0.25],   # sad 悲伤
    [0.4, 0.8, 0.3, 0.7, 0.6, 0.95]     # surprised 惊喜
]
VAD_RANGES = [[v_min * 10, v_max * 10, a_min * 10, a_max * 10, d_min * 10, d_max * 10] for
              (v_min, v_max, a_min, a_max, d_min, d_max) in VAD_RANGES_01]

VAD_MIN = 0.0
VAD_MAX = 10.0
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自监督训练参数
SSL_EPOCHS = 1
SSL_BATCH_SIZE = 16
SSL_LR = 1e-5
TRAIN_EVERY_N_FRAMES = 30
MAX_POOL_SIZE = 200
EVAL_SPLIT_RATIO = 0.4

# MTCNN人脸检测器
mtcnn_detector = MTCNN()

# 推理全局平滑缓存
SSL_PROB_HISTORY = []

# 预处理
classify_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

vad_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ------------------- CBAM + ResNet18 模型定义 -------------------
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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
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


def create_resnet18_with_cbam_regression(output_dim=3, pretrained=False):
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


def create_resnet18_with_cbam_classify(num_classes=7, pretrained=True):
    if pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.cbam = CBAM(in_channels=in_features, reduction=16)
    model.fc = nn.Linear(in_features, num_classes)

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


def load_vad_model(model_path):
    model = create_resnet18_with_cbam_regression(output_dim=3, pretrained=False)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_classify_model():
    model = create_resnet18_with_cbam_classify(num_classes=NUM_CLASSES, pretrained=True)
    model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print("✅ 表情分类模型加载成功")
    return model


# ===================== 推理时序平滑函数 + 锁定峰值类别 =====================
def get_smooth_ssl_prob(new_prob, orig_prob):
    global SSL_PROB_HISTORY
    SSL_PROB_HISTORY.append(new_prob)
    if len(SSL_PROB_HISTORY) > SMOOTH_WINDOW:
        SSL_PROB_HISTORY.pop(0)
    stack = np.array(SSL_PROB_HISTORY)
    smooth_prob = np.mean(stack, axis=0)

    # 强制最高概率类别和原始模型一致，表情不会跨大类突变
    target_max_cls = np.argmax(orig_prob)
    current_max_cls = np.argmax(smooth_prob)
    if target_max_cls != current_max_cls:
        # 提升原始模型对应类别的权重，重新归一化
        smooth_prob[target_max_cls] *= 1.3
        smooth_prob = smooth_prob / np.sum(smooth_prob)
    return smooth_prob


# ===================== 全局时序平滑伪标签（替换原分段平滑） =====================
def generate_pseudo_labels_global(static_model, face_images, frame_indices, window=AVG_NEIGHBOR_RANGE):
    static_model.eval()
    all_probs = []
    with torch.no_grad():
        for face in face_images:
            img_tensor = classify_transform(face).unsqueeze(0).to(DEVICE)
            logits = static_model(img_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    all_probs = np.array(all_probs)
    all_probs = np.nan_to_num(all_probs, nan=0.0)

    pseudo_labels = []
    half_win = (window - 1) // 2
    total = len(all_probs)
    for idx in range(total):
        left = max(0, idx - half_win)
        right = min(total, idx + half_win + 1)
        avg_prob = np.mean(all_probs[left:right], axis=0)
        avg_prob = np.nan_to_num(avg_prob, nan=0.0)
        pseudo_labels.append(avg_prob)
    return np.array(pseudo_labels)


# ===================== 修复时序损失：仅batch内相邻样本计算，消除维度报错 =====================
def self_supervised_train(model, train_faces, frame_ids, pseudo_labels, epochs, batch_size, lr, temporal_weight):
    if lr == 0:
        return model, 0.0
    tensors = []
    for img in train_faces:
        tensors.append(classify_transform(img))
    X = torch.stack(tensors).to(DEVICE)
    y_np = np.nan_to_num(pseudo_labels, nan=0.0)
    y = torch.tensor(y_np, dtype=torch.float32).to(DEVICE)
    idx_tensor = torch.arange(len(X)).to(DEVICE)
    dataset = TensorDataset(X, y, idx_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def kl_loss(pred_logits, target_probs):
        pred_log_probs = F.log_softmax(pred_logits, dim=1)
        target_probs = torch.clamp(target_probs, min=1e-8)
        return F.kl_div(pred_log_probs, target_probs, reduction='batchmean')

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y, batch_idx in loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            pred_prob = F.softmax(pred, dim=1)
            loss_kl = kl_loss(pred, batch_y)

            # 修复：只在当前batch内部相邻帧计算时序损失，不会出现维度不匹配
            loss_temporal = torch.tensor(0., device=DEVICE)
            bsz = pred_prob.shape[0]
            if bsz >= 2:
                loss_temporal = F.mse_loss(pred_prob[1:], pred_prob[:-1])

            total_loss = loss_kl + temporal_weight * loss_temporal
            if torch.isnan(total_loss):
                total_loss = torch.tensor(0.0, device=DEVICE)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += total_loss.item() * batch_X.size(0)

    avg_loss = epoch_loss / len(dataset)
    avg_loss = 0.0 if np.isnan(avg_loss) else avg_loss
    model.eval()
    return model, avg_loss


# ===================== 绘制双线跳变曲线 =====================
def draw_jump_curve(frame, jump_orig, jump_ssl, max_len=200):
    h, w = frame.shape[:2]
    curve_width = 480
    curve_height = 240
    margin = 20
    total_width = w + curve_width + 40
    new_img = np.ones((h, total_width, 3), dtype=np.uint8) * 255
    new_img[0:h, 0:w] = frame.copy()

    if len(jump_orig) == 0:
        return new_img

    fig, ax = plt.subplots(figsize=(curve_width / 100, curve_height / 100), dpi=120)
    orig_clean = np.nan_to_num(jump_orig, nan=0.0)
    ssl_clean = np.nan_to_num(jump_ssl, nan=0.0)
    ax.plot(orig_clean, color='blue', linewidth=1.2, label='Original Model Jump')
    ax.plot(ssl_clean, color='orange', linewidth=1.2, label='SSL Model Jump')
    ax.set_title('Frame-to-Frame Prediction Jump (KL Divergence)', fontsize=9)
    ax.set_xlabel('Frame Index', fontsize=7)
    ax.set_ylabel('KL Divergence', fontsize=7)
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(True, alpha=0.3)

    if len(orig_clean) > max_len:
        ax.set_xlim(len(orig_clean) - max_len, len(orig_clean))

    canvas = FigureCanvas(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    plt.close(fig)

    x_start = w + 20
    y_start = 20
    plot_img = cv2.resize(plot_img, (curve_width, curve_height))
    plot_h, plot_w = plot_img.shape[:2]
    new_img[y_start:y_start + plot_h, x_start:x_start + plot_w] = plot_img
    return new_img


# ---------------- 修复完成的保存曲线图函数（无任何报错，两条曲线分开绘制） ----------------
def save_final_jump_plot(jump_orig, jump_ssl, save_path):
    # 统一转为一维数组，杜绝空列表/标量问题
    arr_orig = np.array(jump_orig, dtype=np.float32).flatten()
    arr_ssl = np.array(jump_ssl, dtype=np.float32).flatten()

    orig_clean = np.nan_to_num(arr_orig, nan=0.0)
    ssl_clean = np.nan_to_num(arr_ssl, nan=0.0)

    # 获取长度用数组shape，安全不会报错
    len_o = orig_clean.shape[0]
    len_s = ssl_clean.shape[0]

    mean_o = float(np.mean(orig_clean)) if len_o > 0 else 0.0
    mean_s = float(np.mean(ssl_clean)) if len_s > 0 else 0.0

    print(f"原始模型跳变均值: {mean_o:.6f}, 总帧数:{len_o}")
    print(f"SSL模型跳变均值: {mean_s:.6f}, 总帧数:{len_s}")

    plt.figure(figsize=(12, 6), dpi=150)
    # 蓝色实线：原始模型
    plt.plot(orig_clean, c='blue', lw=1.4, label='Original Model Jump', ls='-')
    # 红色虚线：SSL模型，和原图区分开，不会重叠看不见
    plt.plot(ssl_clean, c='red', lw=1.4, label='SSL Model Jump', ls='--')

    plt.title("Frame-to-Frame Prediction Jump Comparison")
    plt.xlabel("Frame Index")
    plt.ylabel("KL Divergence")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"✅ 跳变曲线图已保存至: {save_path}")


# ===================== 辅助函数 =====================
def predict_vad_raw(face_img, vad_model, vad_transform, device):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_tensor = vad_transform(face_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        vad = vad_model(img_tensor).cpu().numpy()[0]
    vad = np.clip(vad, 0.0, 10.0)
    vad = np.nan_to_num(vad, nan=0.0)
    return vad


def map_vad_to_emotion_range(vad_raw, emotion_idx, VAD_RANGES):
    ranges = VAD_RANGES[emotion_idx]
    def map_val(x, low, high):
        return low + (x / 10.0) * (high - low)
    v = map_val(vad_raw[0], ranges[0], ranges[1])
    a = map_val(vad_raw[1], ranges[2], ranges[3])
    d = map_val(vad_raw[2], ranges[4], ranges[5])
    res = np.array([v, a, d])
    return np.nan_to_num(res, nan=0.0)


# =====================【修改后的绘图函数】双面板：原始模型 + SSL平滑模型 =====================
def draw_charts(frame,
                vad_ssl, prob_ssl,
                vad_orig, prob_orig,
                class_names_cn, VAD_RANGES, VAD_MIN, VAD_MAX):
    h, w = frame.shape[:2]
    vad_bar_width = 70
    vad_bar_height = 260
    margin = 30
    single_vad_total = 3 * vad_bar_width + 2 * margin

    emotion_bar_width = 200
    emotion_bar_height = 26
    emotion_spacing = 8

    # 两块面板间距
    panel_gap = 60
    # 起始X：原图右侧留白
    panel_start_x = w + 30

    # 总画布宽度扩容，容纳两套VAD+情绪条
    total_panel_width = single_vad_total + panel_gap + single_vad_total
    total_width = panel_start_x + total_panel_width + emotion_bar_width + 60
    new_img = np.ones((h, total_width, 3), dtype=np.uint8) * 255
    new_img[0:h, 0:w] = frame.copy()

    pil_img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        font_title = ImageFont.truetype(font_path, 20)
        font_label = ImageFont.truetype(font_path, 18)
        font_value = ImageFont.truetype(font_path, 16)
        font_small = ImageFont.truetype(font_path, 14)
    except:
        font_title = font_label = font_value = font_small = ImageFont.load_default()

    vad_labels = ["愉悦度V", "激活度A", "支配度D"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    emotion_colors = [
        (255, 0, 0),      # 愤怒 anger
        (0, 128, 0),      # 厌恶 disgust
        (128, 0, 128),    # 惧怕 fear
        (255, 255, 0),    # 开心 happy
        (128, 128, 128),  # 平淡 normal
        (0, 0, 255),      # 悲伤 sad
        (255, 165, 0)     # 惊喜 surprised
    ]

    # ---------------------- 第一块：原始模型 Original ----------------------
    x_offset = panel_start_x
    draw.text((x_offset, 10), "【原始模型输出】", fill=(128,0,0), font=font_title)
    vad_o = np.nan_to_num(vad_orig, nan=0.0)
    vad_o_display = [vad_o[1], vad_o[0], vad_o[2]]
    for i, (val, label, color) in enumerate(zip(vad_o_display, vad_labels, colors)):
        ratio = (val - VAD_MIN) / max((VAD_MAX - VAD_MIN), 1e-8)
        bar_h = int(ratio * vad_bar_height)
        bar_h = max(0, min(bar_h, vad_bar_height))
        x = x_offset + i * (vad_bar_width + margin)
        y_bottom = h - 40
        y_top = y_bottom - bar_h
        draw.line([(x, y_bottom), (x + vad_bar_width, y_bottom)], fill=(200, 200, 200), width=2)
        if bar_h > 0:
            draw.rectangle([(x, y_top), (x + vad_bar_width, y_bottom)], fill=color, outline=(0,0,0))
        draw.text((x, y_top - 18), f"{val:.2f}", fill=(0,0,0), font=font_value)
        draw.text((x, y_bottom + 8), label, fill=(0,0,0), font=font_small)

    # ---------------------- 第二块：SSL时序平滑模型 ----------------------
    x_offset = panel_start_x + single_vad_total + panel_gap
    draw.text((x_offset, 10), "【SSL平滑后输出】", fill=(0,0,128), font=font_title)
    vad_s = np.nan_to_num(vad_ssl, nan=0.0)
    vad_s_display = [vad_s[1], vad_s[0], vad_s[2]]
    for i, (val, label, color) in enumerate(zip(vad_s_display, vad_labels, colors)):
        ratio = (val - VAD_MIN) / max((VAD_MAX - VAD_MIN), 1e-8)
        bar_h = int(ratio * vad_bar_height)
        bar_h = max(0, min(bar_h, vad_bar_height))
        x = x_offset + i * (vad_bar_width + margin)
        y_bottom = h - 40
        y_top = y_bottom - bar_h
        draw.line([(x, y_bottom), (x + vad_bar_width, y_bottom)], fill=(200, 200, 200), width=2)
        if bar_h > 0:
            draw.rectangle([(x, y_top), (x + vad_bar_width, y_bottom)], fill=color, outline=(0,0,0))
        draw.text((x, y_top - 18), f"{val:.2f}", fill=(0,0,0), font=font_value)
        draw.text((x, y_bottom + 8), label, fill=(0,0,0), font=font_small)

    # ---------------------- 右侧：两套情绪概率条形图 ----------------------
    bar_start_x = panel_start_x + total_panel_width + 30
    y_pos = 20
    # 原始模型情绪条
    draw.text((bar_start_x, y_pos), "原始模型情绪权重", fill=(128,0,0), font=font_title)
    y_pos += 30
    prob_o = np.nan_to_num(prob_orig, nan=0.0)
    for name_cn, prob, color in zip(class_names_cn, prob_o, emotion_colors):
        draw.rectangle([(bar_start_x, y_pos), (bar_start_x + emotion_bar_width, y_pos + emotion_bar_height)],
                       fill=(230,230,230), outline=(0,0,0))
        fill_w = int(prob * emotion_bar_width)
        if fill_w > 0:
            draw.rectangle([(bar_start_x, y_pos), (bar_start_x + fill_w, y_pos + emotion_bar_height)],
                           fill=color, outline=(0,0,0))
        draw.text((bar_start_x + 4, y_pos + 2), f"{name_cn} {prob*100:.1f}%", fill=(0,0,0), font=font_small)
        y_pos += emotion_bar_height + emotion_spacing

    # SSL平滑情绪条
    y_pos += 20
    draw.text((bar_start_x, y_pos), "SSL平滑情绪权重", fill=(0,0,128), font=font_title)
    y_pos += 30
    prob_s = np.nan_to_num(prob_ssl, nan=0.0)
    for name_cn, prob, color in zip(class_names_cn, prob_s, emotion_colors):
        draw.rectangle([(bar_start_x, y_pos), (bar_start_x + emotion_bar_width, y_pos + emotion_bar_height)],
                       fill=(230,230,230), outline=(0,0,0))
        fill_w = int(prob * emotion_bar_width)
        if fill_w > 0:
            draw.rectangle([(bar_start_x, y_pos), (bar_start_x + fill_w, y_pos + emotion_bar_height)],
                           fill=color, outline=(0,0,0))
        draw.text((bar_start_x + 4, y_pos + 2), f"{name_cn} {prob*100:.1f}%", fill=(0,0,0), font=font_small)
        y_pos += emotion_bar_height + emotion_spacing

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result


# ===================== 主程序 =====================
def process_video():
    # ========== 修复1：提前创建输出全部文件夹 ==========
    out_dir = os.path.dirname(VIDEO_OUTPUT_PATH)
    curve_dir = os.path.dirname(JUMP_CURVE_SAVE_PATH)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    print(f"输出视频目录已确认: {out_dir}")
    print(f"曲线图保存目录已确认: {curve_dir}")

    original_classify_model = load_classify_model()
    ssl_model = load_classify_model()
    vad_model = load_vad_model(VAD_MODEL_PATH)
    print(f"✅ 模型加载完成，平滑窗口帧数：{AVG_NEIGHBOR_RANGE}，SSL学习率：{SSL_LR}")

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"❌ 无法打开输入视频: {VIDEO_INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"视频信息: {orig_w}x{orig_h}, {fps:.2f} fps")

    prev_probs_orig = None
    prev_probs_ssl = None
    jump_values_orig = []
    jump_values_ssl = []

    train_pool = {"faces": [], "frame_ids": [], "seg_ids": [], "seg_idx": []}
    eval_pool = {"faces": [], "frame_ids": [], "seg_ids": [], "seg_idx": []}
    last_frame_idx = -1
    seg_counter = 0
    train_seg_id = 0
    eval_seg_id = 0
    train_counter = 0

    # ========== 修复2：缩减侧边宽度，降低宽高比，改用H264编码avc1 ==========
    extra_side_width = 800   # 原1200 → 800，画面不会过宽
    out_w = orig_w + extra_side_width
    out_h = orig_h
    # 优先avc1(H.264)，失败再 fallback mp4v
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        print("avc1编码初始化失败，切换mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        print(f"❌ 视频写入器初始化失败！路径：{VIDEO_OUTPUT_PATH} 分辨率：{out_w}×{out_h}")
        cap.release()
        return

    print(f"✅ VideoWriter初始化成功，输出尺寸 {out_w} × {out_h}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mtcnn_detector.detect_faces(rgb_frame)
        # SSL结果
        emotion_probs_ssl = np.zeros(NUM_CLASSES)
        vad_values_ssl = [0.0, 0.0, 0.0]
        # 原始模型结果
        emotion_probs_orig = np.zeros(NUM_CLASSES)
        vad_values_orig = [0.0, 0.0, 0.0]
        jump_orig = 0.0
        jump_ssl = 0.0

        if len(results) > 0:
            best_face = max(results, key=lambda x: x["confidence"])
            x1, y1, w_box, h_box = best_face["box"]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x1 + w_box)
            y2 = min(frame.shape[0], y1 + h_box)
            face_roi = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img_tensor = classify_transform(face_roi).unsqueeze(0).to(DEVICE)
            # 原始模型预测
            with torch.no_grad():
                logits_orig = original_classify_model(img_tensor)
                probs_orig = F.softmax(logits_orig, dim=1).cpu().numpy()[0]
            probs_orig = np.nan_to_num(probs_orig, nan=0.0)
            emotion_probs_orig = probs_orig

            # SSL模型预测 + 时序平滑 + 锁定峰值类别
            with torch.no_grad():
                logits_ssl = ssl_model(img_tensor)
                probs_ssl_raw = F.softmax(logits_ssl, dim=1).cpu().numpy()[0]
            probs_ssl_raw = np.nan_to_num(probs_ssl_raw, nan=0.0)
            probs_ssl = get_smooth_ssl_prob(probs_ssl_raw, probs_orig)
            emotion_probs_ssl = probs_ssl

            # 计算KL跳变
            eps = 1e-8
            if prev_probs_orig is not None:
                p_orig = np.clip(prev_probs_orig, eps, 1.0)
                q_orig = np.clip(probs_orig, eps, 1.0)
                kl_orig = np.sum(p_orig * np.log(p_orig / q_orig))
                jump_orig = max(0.0, kl_orig)
            else:
                jump_orig = 0.0

            if prev_probs_ssl is not None:
                p_ssl = np.clip(prev_probs_ssl, eps, 1.0)
                q_ssl = np.clip(probs_ssl, eps, 1.0)
                kl_ssl = np.sum(p_ssl * np.log(p_ssl / q_ssl))
                jump_ssl = max(0.0, kl_ssl)
            else:
                jump_ssl = 0.0

            prev_probs_orig = probs_orig.copy()
            prev_probs_ssl = probs_ssl.copy()

            # VAD 两套映射（分别用各自预测的主导情绪）
            vad_raw = predict_vad_raw(face_roi, vad_model, vad_transform, DEVICE)
            # 原始模型VAD
            dom_orig = np.argmax(probs_orig)
            vad_mapped_orig = map_vad_to_emotion_range(vad_raw, dom_orig, VAD_RANGES)
            vad_values_orig = vad_mapped_orig.tolist()
            # SSL平滑模型VAD
            dom_ssl = np.argmax(probs_ssl)
            vad_mapped_ssl = map_vad_to_emotion_range(vad_raw, dom_ssl, VAD_RANGES)
            vad_values_ssl = vad_mapped_ssl.tolist()

            # 收集eval池
            current_frame_idx = frame_idx
            if len(eval_pool["faces"]) > 0 and (current_frame_idx - last_frame_idx) == 1:
                seg_counter += 1
                within_idx = seg_counter
                seg_id = eval_seg_id
            else:
                eval_seg_id += 1
                seg_id = eval_seg_id
                seg_counter = 0
                within_idx = 0

            eval_pool["faces"].append(face_roi)
            eval_pool["frame_ids"].append(current_frame_idx)
            eval_pool["seg_ids"].append(seg_id)
            eval_pool["seg_idx"].append(within_idx)
            last_frame_idx = current_frame_idx

            if len(eval_pool["faces"]) > MAX_POOL_SIZE:
                for key in eval_pool.keys():
                    eval_pool[key].pop(0)

            # 定时自监督训练
            if len(eval_pool["faces"]) >= 3 and (train_counter % TRAIN_EVERY_N_FRAMES == 0):
                total_eval_len = len(eval_pool["faces"])
                split_point = int(total_eval_len * (1 - EVAL_SPLIT_RATIO))
                train_faces = eval_pool["faces"][:split_point]
                train_fids = eval_pool["frame_ids"][:split_point]

                eval_faces = eval_pool["faces"][split_point:]
                eval_fids = eval_pool["frame_ids"][split_point:]
                eval_segs = eval_pool["seg_ids"][split_point:]
                eval_segidx = eval_pool["seg_idx"][split_point:]

                eval_pool["faces"] = eval_faces
                eval_pool["frame_ids"] = eval_fids
                eval_pool["seg_ids"] = eval_segs
                eval_pool["seg_idx"] = eval_segidx

                train_seg_id += 1
                train_pool["faces"].extend(train_faces)
                train_pool["frame_ids"].extend(train_fids)
                train_pool["seg_ids"].extend([train_seg_id] * len(train_faces))
                train_pool["seg_idx"].extend(list(range(len(train_faces))))

                if len(train_pool["faces"]) > MAX_POOL_SIZE:
                    pop_num = len(train_pool["faces"]) - MAX_POOL_SIZE
                    for key in train_pool.keys():
                        del train_pool[key][:pop_num]

                # 全局平滑伪标签
                train_pseudo = generate_pseudo_labels_global(original_classify_model, train_pool["faces"], train_pool["frame_ids"])
                # 带时序损失训练
                ssl_model, _ = self_supervised_train(
                    ssl_model,
                    train_pool["faces"],
                    train_pool["frame_ids"],
                    train_pseudo,
                    SSL_EPOCHS,
                    SSL_BATCH_SIZE,
                    SSL_LR,
                    temporal_weight=TEMPORAL_LOSS_WEIGHT
                )

                # 滑动窗口淘汰，不清空eval池
                max_eval_size = MAX_POOL_SIZE // 2
                if len(eval_pool["faces"]) > max_eval_size:
                    pop_n = len(eval_pool["faces"]) - max_eval_size
                    for key in eval_pool.keys():
                        del eval_pool[key][:pop_n]
                seg_counter = 0

        jump_values_orig.append(jump_orig)
        jump_values_ssl.append(jump_ssl)
        train_counter += 1

        # 传入两套VAD、两套概率给绘图函数
        display_frame = draw_charts(
            frame,
            vad_values_ssl, emotion_probs_ssl,
            vad_values_orig, emotion_probs_orig,
            CLASS_NAMES_CN, VAD_RANGES, VAD_MIN, VAD_MAX
        )
        display_frame = draw_jump_curve(display_frame, jump_values_orig, jump_values_ssl, max_len=200)

        # ========== 修复3：强制统一尺寸，防止shape不匹配写入失败 ==========
        if display_frame.shape[0] != out_h or display_frame.shape[1] != out_w:
            display_frame = cv2.resize(display_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.write(display_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧")

    cap.release()
    out.release()
    save_final_jump_plot(jump_values_orig, jump_values_ssl, JUMP_CURVE_SAVE_PATH)
    print(f"分析完成！视频已保存至: {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    process_video()