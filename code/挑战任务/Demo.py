# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from PIL import ImageFont, ImageDraw, Image
import streamlit as st
import io

# ===================== 全局配置（相对路径，交付老师无需改盘符） =====================
CLASS_MODEL_PATH = "../../results-model/model1/best_model_ALL.pth"
VAD_MODEL_PATH = "../../results-model/model2/best_model_Emotic_VAD_balanced.pth"
SAVE_FOLDER = "../../pred_output"

NUM_CLASSES = 7
ALPHA = 0.25  # 时序平滑系数，和本地代码统一
CLASS_NAMES_CN = ["愤怒", "厌恶", "惧怕", "开心", "平淡", "悲伤", "惊喜"]

VAD_RANGES_01 = [
    [0.7, 1.0, 0.15, 0.4, 0.7, 1.0],    # 愤怒
    [0.3, 0.7, 0.2, 0.45, 0.2, 0.6],    # 厌恶
    [0.5, 0.9, 0.1, 0.4, 0.2, 0.5],     # 惧怕
    [0.3, 0.9, 0.8, 1.0, 0.6, 0.85],    # 开心
    [0.0, 0.2, 0.4, 0.6, 0.0, 0.4],     # 平淡
    [0.4, 0.85, 0.0, 0.2, 0.0, 0.25],   # 悲伤
    [0.4, 0.8, 0.3, 0.7, 0.6, 0.95]     # 惊喜
]
VAD_RANGES = [[v_min * 10, v_max * 10, a_min * 10, a_max * 10, d_min * 10, d_max * 10] for
              (v_min, v_max, a_min, a_max, d_min, d_max) in VAD_RANGES_01]

VAD_MIN = 0.0
VAD_MAX = 10.0
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 图像预处理
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

# ------------------- CBAM 模型模块（完全复用你原有代码） -------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
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
        return self.sigmoid(avg_out + max_out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat)) * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def create_resnet18_with_cbam_regression(output_dim=3, pretrained=False):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.cbam = CBAM(in_features, 16)
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
        return model.fc(x)
    model.forward = new_forward
    return model

def create_resnet18_with_cbam_classify(num_classes=7, pretrained=True):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    in_features = model.fc.in_features
    model.cbam = CBAM(in_features, 16)
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
        return model.fc(x)
    model.forward = new_forward
    return model

def load_vad_model(model_path):
    model = create_resnet18_with_cbam_regression(3, False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_classify_model():
    model = create_resnet18_with_cbam_classify(NUM_CLASSES, True)
    model.load_state_dict(torch.load(CLASS_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model

# 时序平滑（和你本地摄像头代码逻辑完全一致）
def get_smooth_prob(new_prob, prev_smooth):
    if prev_smooth is None:
        return new_prob.copy()
    smooth = ALPHA * new_prob + (1 - ALPHA) * prev_smooth
    smooth = smooth / np.sum(smooth)
    return smooth

# VAD计算
def predict_vad_raw(face_img, vad_model, device):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    tensor = vad_transform(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        vad = vad_model(tensor).cpu().numpy()[0]
    return np.clip(vad, 0, 10)

def map_vad_to_emotion_range(vad_raw, emo_idx):
    v_min, v_max, a_min, a_max, d_min, d_max = VAD_RANGES[emo_idx]
    v = v_min + (vad_raw[0]/10)*(v_max - v_min)
    a = a_min + (vad_raw[1]/10)*(a_max - a_min)
    d = d_min + (vad_raw[2]/10)*(d_max - d_min)
    return np.array([v,a,d])

# 绘图函数（补齐全部参数，无报错）
def draw_charts(frame, vad_values, emotion_probs):
    h, w = frame.shape[:2]
    vad_bar_w = 80
    vad_bar_h = 280
    margin = 40
    vad_start_x = w + 40
    emo_bar_w = 220
    emo_bar_h = 28
    emo_gap = 10
    emo_start_x = vad_start_x + 3*vad_bar_w + 2*margin + 40
    emo_start_y = 60
    total_w = emo_start_x + emo_bar_w + 80

    canvas = np.ones((h, total_w, 3), dtype=np.uint8)*255
    canvas[:h, :w] = frame.copy()
    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 22)
        font_txt = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
    except:
        font_title = font_txt = ImageFont.load_default()

    vad_display = [vad_values[1], vad_values[0], vad_values[2]]
    vad_labels = ["愉悦度", "情感度", "主动度"]
    vad_colors = [(0,0,255), (0,255,0), (255,0,0)]
    for i, val in enumerate(vad_display):
        ratio = val / 10
        bar_h = int(ratio * vad_bar_h)
        x = vad_start_x + i*(vad_bar_w + margin)
        y_bottom = h - 50
        y_top = y_bottom - bar_h
        draw.rectangle([x, y_top, x+vad_bar_w, y_bottom], fill=vad_colors[i], outline=(0,0,0))
        draw.text((x, y_top-20), f"{val:.2f}", font=font_txt, fill=(0,0,0))
        draw.text((x, y_bottom+10), vad_labels[i], font=font_txt, fill=(0,0,0))
    draw.text((vad_start_x, 30), "AVD三维情感指标", font=font_title, fill=(0,0,0))

    emo_colors = [(255,0,0),(0,128,0),(128,0,128),(255,255,0),(128,128,128),(0,0,255),(255,165,0)]
    draw.text((emo_start_x, emo_start_y-25), "7类情绪概率分布", font=font_title, fill=(0,0,0))
    for i, (name, prob, color) in enumerate(zip(CLASS_NAMES_CN, emotion_probs, emo_colors)):
        y = emo_start_y + i*(emo_bar_h + emo_gap)
        draw.rectangle([emo_start_x, y, emo_start_x+emo_bar_w, y+emo_bar_h], fill=(230,230,230), outline=(0,0,0))
        fill_w = int(prob * emo_bar_w)
        draw.rectangle([emo_start_x, y, emo_start_x+fill_w, y+emo_bar_h], fill=color)
        draw.text((emo_start_x+5, y+3), f"{name} {prob*100:.1f}%", font=font_txt, fill=(0,0,0))
    return cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)

# ---------------- Streamlit 网页主逻辑 ----------------
if __name__ == "__main__":
    st.set_page_config(page_title="网页实时人脸情感识别", layout="wide")
    st.title("多模态人脸情感识别可视化Demo")
    st.markdown("""
    ✅ 实时监控：开启自动刷新，网页持续摄像头实时识别（和本地OpenCV逻辑一致）
    ✅ 水平镜像开关：前置摄像头画面左右翻转，解决镜像别扭问题
    ✅ 保存规则：仅上传视频自动存入本地pred_output，图片/实时摄像头仅网页展示
    """)
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # 全局缓存模型，只加载一次
    @st.cache_resource
    def load_models():
        with st.spinner("正在加载双模型，请稍候..."):
            cls_m = load_classify_model()
            vad_m = load_vad_model(VAD_MODEL_PATH)
        st.success("✅ 表情分类+VAD回归模型加载完成")
        return cls_m, vad_m
    cls_model, vad_model = load_models()

    # 控制开关
    mirror_toggle = st.checkbox("开启水平镜像（修正前置摄像头反向）", value=True)
    realtime_monitor = st.checkbox("开启网页实时情感监控（自动刷新画面）", value=True)

    col_left, col_right = st.columns([1, 1.2])
    with col_left:
        upload_img = st.file_uploader("上传静态图片", type=["jpg","png","jpeg"])
        upload_vid = st.file_uploader("上传视频（自动本地保存结果）", type=["mp4","avi","mov"])
        cam_input = st.camera_input("摄像头实时画面")

    # 右侧结果展示区域
    res_holder = col_right.empty()
    download_btn = col_right.empty()
    smooth_state = None

    # 统一推理渲染函数
    def run_inference(pil_origin):
        nonlocal smooth_state
        # 镜像翻转（等同于cv2.flip(frame,1)，全局生效）
        if mirror_toggle:
            pil_origin = pil_origin.transpose(Image.FLIP_LEFT_RIGHT)
        frame_bgr = cv2.cvtColor(np.array(pil_origin), cv2.COLOR_RGB2BGR)
        emo_prob = np.zeros(NUM_CLASSES)
        vad_out = [0.0,0.0,0.0]

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        if len(faces) > 0:
            # 取最大人脸
            x,y,w,h = max(faces, key=lambda box: box[2]*box[3])
            cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), (0,255,0), 2)
            face_roi = frame_bgr[y:y+h, x:x+w]
            # 分类推理+时序平滑
            tensor_cls = classify_transform(face_roi).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = cls_model(tensor_cls)
                raw_prob = F.softmax(logits, dim=1).cpu().numpy()[0]
            smooth_state = get_smooth_prob(raw_prob, smooth_state)
            emo_prob = smooth_state
            # VAD回归
            vad_raw = predict_vad_raw(face_roi, vad_model, DEVICE)
            main_emo = np.argmax(emo_prob)
            vad_out = map_vad_to_emotion_range(vad_raw, main_emo).tolist()
        # 绘制图表
        out_bgr = draw_charts(frame_bgr, vad_out, emo_prob)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        res_img = Image.fromarray(out_rgb)
        # 网页展示
        res_holder.image(res_img, caption="实时情感识别结果", use_column_width=True)
        # 浏览器下载按钮
        buf = io.BytesIO()
        res_img.save(buf, format="PNG")
        download_btn.download_button("浏览器下载当前画面", buf.getvalue(), f"emo_{time.time():.0f}.png")
        return res_img

    # 1. 摄像头画面处理（实时核心）
    if cam_input:
        img_pil = Image.open(cam_input).convert("RGB")
        run_inference(img_pil)
        # 实时监控开启则自动刷新页面，持续获取新帧
        if realtime_monitor:
            time.sleep(0.7)
            st.rerun()

    # 2. 静态图片上传
    if upload_img:
        img_pil = Image.open(upload_img).convert("RGB")
        run_inference(img_pil)

    # 3. 视频上传（仅视频自动保存本地）
    if upload_vid:
        temp_path = "temp_video_cache.mp4"
        with open(temp_path, "wb") as f:
            f.write(upload_vid.read())
        cap = cv2.VideoCapture(temp_path)
        ok, frame_bgr = cap.read()
        cap.release()
        os.remove(temp_path)
        if not ok:
            st.error("视频读取失败")
        else:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_vid = Image.fromarray(frame_rgb)
            res_vid = run_inference(pil_vid)
            # 自动保存到本地pred_output
            save_name = f"video_result_{time.strftime('%Y%m%d_%H%M%S')}.png"
            full_save = os.path.join(SAVE_FOLDER, save_name)
            res_vid.save(full_save)
            col_right.success(f"视频分析图已自动保存：{full_save}")