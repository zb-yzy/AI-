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

# ===================== 配置 =====================
CLASS_MODEL_PATH = "../../results-model/model1/best_model_ALL.pth"
VAD_MODEL_PATH = "../../results-model/model2/best_model_Emotic_VAD_balanced.pth"

VIDEO_OUTPUT_PATH = "../../results/vidio/camera/camera_adjusted_analyzed.mp4"

NUM_CLASSES = 7
# 指数加权系数，新帧权重0.25，旧帧权重持续衰减
ALPHA = 0.25

CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
CLASS_NAMES_CN = ["愤怒", "厌恶", "惧怕", "开心", "平淡", "悲伤", "惊喜"]

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

# OpenCV Haar人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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


# 指数加权平滑：越旧帧权重越小
smooth_prev = None
ALPHA = 0.25

def get_smooth_prob(new_prob):
    global smooth_prev
    if smooth_prev is None:
        smooth_prev = new_prob.copy()
        return new_prob.copy()
    # 新帧权重ALPHA，历史累积权重(1-ALPHA)，旧帧持续衰减
    smooth_new = ALPHA * new_prob + (1 - ALPHA) * smooth_prev
    smooth_new = smooth_new / np.sum(smooth_new)
    smooth_prev = smooth_new.copy()
    return smooth_new


# ===================== 辅助绘图函数 =====================
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


def draw_charts(frame, vad_values, emotion_probs, class_names_cn, VAD_RANGES, VAD_MIN, VAD_MAX):
    h, w = frame.shape[:2]
    vad_bar_width = 80
    vad_bar_height = 280
    margin = 40
    total_vad_width = 3 * vad_bar_width + 2 * margin
    start_vad_x = w + 40

    emotion_bar_width = 220
    emotion_bar_height = 28
    emotion_spacing = 10
    start_emotion_x = start_vad_x + total_vad_width + 40
    start_emotion_y = 60

    total_width = start_emotion_x + emotion_bar_width + 80
    new_img = np.ones((h, total_width, 3), dtype=np.uint8) * 255
    new_img[0:h, 0:w] = frame.copy()

    pil_img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        font_title = ImageFont.truetype(font_path, 22)
        font_label = ImageFont.truetype(font_path, 20)
        font_value = ImageFont.truetype(font_path, 18)
        font_small = ImageFont.truetype(font_path, 16)
    except:
        font_title = font_label = font_value = font_small = ImageFont.load_default()

    vad_values = np.nan_to_num(vad_values, nan=0.0)
    vad_display = [vad_values[1], vad_values[0], vad_values[2]]
    vad_labels = ["愉悦度", "情感度", "主动度"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i, (val, label, color) in enumerate(zip(vad_display, vad_labels, colors)):
        ratio = (val - VAD_MIN) / max((VAD_MAX - VAD_MIN), 1e-8)
        bar_h = int(ratio * vad_bar_height)
        bar_h = max(0, min(bar_h, vad_bar_height))
        x = start_vad_x + i * (vad_bar_width + margin)
        y_bottom = h - 50
        y_top = y_bottom - bar_h
        draw.line([(x, y_bottom), (x + vad_bar_width, y_bottom)], fill=(200, 200, 200), width=2)
        if bar_h > 0:
            draw.rectangle([(x, y_top), (x + vad_bar_width, y_bottom)], fill=color, outline=(0, 0, 0))
        else:
            draw.rectangle([(x, y_bottom - 2), (x + vad_bar_width, y_bottom)], fill=color, outline=(0, 0, 0))
        draw.text((x, y_top - 20), f"{val:.2f}", fill=(0, 0, 0), font=font_value)
        draw.text((x, y_bottom + 10), label, fill=(0, 0, 0), font=font_label)
    draw.text((start_vad_x, 30), "映射后 VAD (AVD)", fill=(0, 0, 0), font=font_title)

    draw.text((start_emotion_x, start_emotion_y - 25), "情绪权重分布", fill=(0, 0, 0), font=font_title)
    emotion_colors = [
        (255, 0, 0),      # 愤怒 anger
        (0, 128, 0),      # 厌恶 disgust
        (128, 0, 128),    # 惧怕 fear
        (255, 255, 0),    # 开心 happy
        (128, 128, 128),  # 平淡 normal
        (0, 0, 255),      # 悲伤 sad
        (255, 165, 0)     # 惊喜 surprised
    ]
    emotion_probs = np.nan_to_num(emotion_probs, nan=0.0)
    for i, (name_cn, prob, color) in enumerate(zip(class_names_cn, emotion_probs, emotion_colors)):
        y = start_emotion_y + i * (emotion_bar_height + emotion_spacing)
        draw.rectangle([(start_emotion_x, y), (start_emotion_x + emotion_bar_width, y + emotion_bar_height)],
                       fill=(230, 230, 230), outline=(0, 0, 0))
        fill_w = int(prob * emotion_bar_width)
        if fill_w > 0:
            draw.rectangle([(start_emotion_x, y), (start_emotion_x + fill_w, y + emotion_bar_height)],
                           fill=color, outline=(0, 0, 0))
        draw.text((start_emotion_x + 5, y + 3), f"{name_cn} {prob * 100:.1f}%", fill=(0, 0, 0), font=font_small)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return result


# ===================== 主程序 =====================
def process_camera():
    global smooth_prev
    # 每次启动重置平滑缓存
    smooth_prev = None
    os.makedirs(os.path.dirname(VIDEO_OUTPUT_PATH), exist_ok=True)
    original_classify_model = load_classify_model()
    vad_model = load_vad_model(VAD_MODEL_PATH)
    print(f"✅ 模型加载完成，指数平滑系数ALPHA={ALPHA}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查设备！")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {orig_w}x{orig_h}, {fps:.2f} fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    extra_side_width = 720
    out_w = orig_w + extra_side_width
    out_h = orig_h
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        print("❌ 视频写入器初始化失败")
        return

    frame_idx = 0
    print("===== 摄像头实时识别启动，按 q 退出 =====")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败，退出")
            break

        # 水平镜像翻转
        frame = cv2.flip(frame, 1)

        emotion_probs = np.zeros(NUM_CLASSES)
        vad_values = [0.0, 0.0, 0.0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            max_area = 0
            best_face = None
            for (x, y, w_box, h_box) in faces:
                area = w_box * h_box
                if area > max_area:
                    max_area = area
                    best_face = (x, y, w_box, h_box)
            x1, y1, w_box, h_box = best_face
            x2 = x1 + w_box
            y2 = y1 + h_box
            face_roi = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            img_tensor = classify_transform(face_roi).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = original_classify_model(img_tensor)
                probs_raw = F.softmax(logits, dim=1).cpu().numpy()[0]
            probs_raw = np.nan_to_num(probs_raw, nan=0.0)
            probs_smooth = get_smooth_prob(probs_raw)
            emotion_probs = probs_smooth

            vad_raw = predict_vad_raw(face_roi, vad_model, vad_transform, DEVICE)
            dominant_idx = np.argmax(emotion_probs)
            vad_mapped = map_vad_to_emotion_range(vad_raw, dominant_idx, VAD_RANGES)
            vad_values = vad_mapped.tolist()

        display_frame = draw_charts(frame, vad_values, emotion_probs, CLASS_NAMES_CN, VAD_RANGES, VAD_MIN, VAD_MAX)
        if display_frame.shape != (out_h, out_w, 3):
            display_frame = cv2.resize(display_frame, (out_w, out_h))

        out.write(display_frame)
        cv2.imshow("Camera Emotion Recognition (Press q to quit)", display_frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"已处理 {frame_idx} 帧")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"识别结束！录制视频保存至: {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    process_camera()