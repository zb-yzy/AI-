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
import tempfile          # 补充导入

# ===================== 全局配置 =====================
CLASS_MODEL_PATH = "results-model/model1/best_model_ALL.pth"
VAD_MODEL_PATH = "results-model/model2/best_model_Emotic_VAD_balanced.pth"
SAVE_FOLDER = "pred_output"
VIDEO_OUTPUT_DIR = "results/vidio/Demo"

NUM_CLASSES = 7
ALPHA = 0.25
CLASS_NAMES_CN = ["愤怒", "厌恶", "惧怕", "开心", "平淡", "悲伤", "惊喜"]

VAD_RANGES_01 = [
    [0.7, 1.0, 0.15, 0.4, 0.7, 1.0],
    [0.3, 0.7, 0.2, 0.45, 0.2, 0.6],
    [0.5, 0.9, 0.1, 0.4, 0.2, 0.5],
    [0.3, 0.9, 0.8, 1.0, 0.6, 0.85],
    [0.0, 0.2, 0.4, 0.6, 0.0, 0.4],
    [0.4, 0.85, 0.0, 0.2, 0.0, 0.25],
    [0.4, 0.8, 0.3, 0.7, 0.6, 0.95]
]
VAD_RANGES = [[v_min * 10, v_max * 10, a_min * 10, a_max * 10, d_min * 10, d_max * 10] for
              (v_min, v_max, a_min, a_max, d_min, d_max) in VAD_RANGES_01]

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
haar_params = {"scaleFactor": 1.05, "minNeighbors": 3, "minSize": (30, 30)}

def detect_faces(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=haar_params["scaleFactor"],
        minNeighbors=haar_params["minNeighbors"],
        minSize=haar_params["minSize"]
    )
    return list(faces)

classify_transform = transforms.Compose([
    transforms.ToPILImage(),
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

# ------------------- CBAM 模型模块 -------------------
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

# 绘图函数（增加高度，确保图表完整显示）
def draw_charts(frame, vad_values, emotion_probs, debug_info=None):
    h, w = frame.shape[:2]
    # 画布高度 = 原始帧高度 + 固定额外区域（400像素），保证图表不重叠
    extra_height = 400
    canvas_h = h + extra_height
    vad_bar_w = 80
    vad_bar_h = 280
    margin = 40
    vad_start_x = w + 40
    emo_bar_w = 220
    emo_bar_h = 28
    emo_gap = 10
    emo_start_x = vad_start_x + 3 * vad_bar_w + 2 * margin + 40
    emo_start_y = 60
    total_w = emo_start_x + emo_bar_w + 80

    canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * 255
    canvas[:h, :w] = frame.copy()

    pil_canvas = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_canvas)
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 22)
        font_txt = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
    except:
        font_title = font_txt = ImageFont.load_default()

    # VAD 柱状图（定位在额外区域底部）
    vad_display = [vad_values[1], vad_values[0], vad_values[2]]
    vad_labels = ["愉悦度", "情感度", "主动度"]
    vad_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    y_bottom_vad = canvas_h - 50
    for i, val in enumerate(vad_display):
        ratio = val / 10.0
        bar_h = int(ratio * vad_bar_h)
        x = vad_start_x + i * (vad_bar_w + margin)
        y_top = y_bottom_vad - bar_h
        draw.rectangle([x, y_top, x + vad_bar_w, y_bottom_vad], fill=vad_colors[i], outline=(0, 0, 0))
        draw.text((x, y_top - 20), f"{val:.2f}", font=font_txt, fill=(0, 0, 0))
        draw.text((x, y_bottom_vad + 10), vad_labels[i], font=font_txt, fill=(0, 0, 0))
    draw.text((vad_start_x, y_bottom_vad - vad_bar_h - 60), "AVD三维情感指标", font=font_title, fill=(0, 0, 0))

    # 情绪概率柱状图（放在原帧下方区域）
    emo_colors = [(255, 0, 0), (0, 128, 0), (128, 0, 128), (255, 255, 0),
                  (128, 128, 128), (0, 0, 255), (255, 165, 0)]
    draw.text((emo_start_x, emo_start_y - 25), "7类情绪概率分布", font=font_title, fill=(0, 0, 0))
    for i, (name, prob, color) in enumerate(zip(CLASS_NAMES_CN, emotion_probs, emo_colors)):
        y = emo_start_y + i * (emo_bar_h + emo_gap)
        draw.rectangle([emo_start_x, y, emo_start_x + emo_bar_w, y + emo_bar_h],
                       fill=(230, 230, 230), outline=(0, 0, 0))
        fill_w = int(prob * emo_bar_w)
        draw.rectangle([emo_start_x, y, emo_start_x + fill_w, y + emo_bar_h], fill=color)
        draw.text((emo_start_x + 5, y + 3), f"{name} {prob * 100:.1f}%", font=font_txt, fill=(0, 0, 0))

    if debug_info:
        debug_y = emo_start_y + len(CLASS_NAMES_CN) * (emo_bar_h + emo_gap) + 20
        for line in debug_info:
            draw.text((emo_start_x, debug_y), line, font=font_txt, fill=(0, 0, 0))
            debug_y += 25

    return cv2.cvtColor(np.array(pil_canvas), cv2.COLOR_RGB2BGR)

# 推理函数（支持跳帧）
def process_frame(frame_bgr, cls_model, vad_model, smooth_cache=None, force_inference=True):
    debug_lines = []
    emo_prob = np.zeros(NUM_CLASSES)
    vad_out = [0.0, 0.0, 0.0]
    face_count = 0

    if not force_inference and "last_emo" in st.session_state and st.session_state.last_emo is not None:
        emo_prob = st.session_state.last_emo
        vad_out = st.session_state.last_vad
        if "last_face_rect" in st.session_state:
            x, y, w, h = st.session_state.last_face_rect
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        out_bgr = draw_charts(frame_bgr, vad_out, emo_prob, ["（跳帧复用上次结果）"])
        return out_bgr, smooth_cache, st.session_state.get("last_face_count", 0)

    faces = detect_faces(frame_bgr)
    face_count = len(faces)
    debug_lines.append(f"人脸检测数: {face_count}")

    if face_count > 0:
        areas = [w*h for (_, _, w, h) in faces]
        idx = np.argmax(areas)
        x, y, w, h = faces[idx]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_roi = frame_bgr[y:y + h, x:x + w]

        tensor_cls = classify_transform(face_roi).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = cls_model(tensor_cls)
            raw_prob = F.softmax(logits, dim=1).cpu().numpy()[0]
        debug_lines.append(f"原始分类概率: {np.array2string(raw_prob, precision=3)}")

        if smooth_cache is not None:
            new_prob = ALPHA * raw_prob + (1 - ALPHA) * smooth_cache
            new_prob = new_prob / np.sum(new_prob)
            smooth_cache = new_prob.copy()
            emo_prob = new_prob
        else:
            emo_prob = raw_prob.copy()

        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        tensor_vad = vad_transform(rgb_face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vad_raw = vad_model(tensor_vad).cpu().numpy()[0]
        vad_raw = np.clip(vad_raw, 0, 10)
        debug_lines.append(f"VAD原始值: {np.array2string(vad_raw, precision=3)}")

        main_emo = np.argmax(emo_prob)
        v_min, v_max, a_min, a_max, d_min, d_max = VAD_RANGES[main_emo]
        v = v_min + (vad_raw[0] / 10) * (v_max - v_min)
        a = a_min + (vad_raw[1] / 10) * (a_max - a_min)
        d = d_min + (vad_raw[2] / 10) * (d_max - d_min)
        vad_out = [v, a, d]

        st.session_state.last_emo = emo_prob
        st.session_state.last_vad = vad_out
        st.session_state.last_face_rect = (x, y, w, h)
        st.session_state.last_face_count = face_count
    else:
        debug_lines.append("未检测到人脸，输出置零")
        st.session_state.last_emo = None

    out_bgr = draw_charts(frame_bgr, vad_out, emo_prob, debug_info=debug_lines)
    return out_bgr, smooth_cache, face_count

# ---------------- Streamlit 主逻辑 ----------------
if __name__ == "__main__":
    st.set_page_config(page_title="网页实时人脸情感识别", layout="wide")
    st.title("多模态人脸情感识别可视化Demo")
    st.markdown("""
    ✅ 实时监控：点击侧边栏按钮启动摄像头，支持跳帧加速  
    ✅ 水平镜像：勾选后画面左右翻转  
    ✅ 权限申请：先点击“申请摄像头权限”按钮让浏览器授权  
    ✅ 完整视频分析：上传视频后逐帧处理，输出带图表的新视频
    """)
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

    # session_state 初始化
    for key in ["mirror_state", "is_monitoring", "camera", "smooth_cache",
                "frame_counter", "skip_frames", "last_emo", "last_vad",
                "last_face_rect", "last_face_count"]:
        if key not in st.session_state:
            st.session_state[key] = None
    if "mirror_state" not in st.session_state: st.session_state.mirror_state = True
    if "is_monitoring" not in st.session_state: st.session_state.is_monitoring = False
    if "skip_frames" not in st.session_state: st.session_state.skip_frames = 2
    if "frame_counter" not in st.session_state: st.session_state.frame_counter = 0

    @st.cache_resource
    def load_models():
        with st.spinner("⏳ 正在加载 AI 模型，请稍候..."):
            cls_m = load_classify_model()
            vad_m = load_vad_model(VAD_MODEL_PATH)
        st.success("✅ 表情分类 + VAD 回归模型已就绪")
        return cls_m, vad_m

    cls_model, vad_model = load_models()

    # ---------- 侧边栏控制 ----------
    st.sidebar.header("控制选项")
    mirror_toggle = st.sidebar.checkbox("开启水平镜像", value=st.session_state["mirror_state"])
    st.session_state["mirror_state"] = mirror_toggle

    skip_slider = st.sidebar.slider("跳帧间隔（1=每帧推理，3=每3帧推理）", 1, 5, st.session_state["skip_frames"])
    st.session_state["skip_frames"] = skip_slider

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔑 摄像头权限")
    if st.sidebar.button("申请摄像头权限"):
        js_code = """
        <script>
        navigator.mediaDevices.getUserMedia({video: true})
            .then(stream => {
                stream.getTracks().forEach(track => track.stop());
                alert('✅ 摄像头权限已允许！可以开始监控');
            })
            .catch(err => {
                if (err.name === 'NotAllowedError') {
                    alert('❌ 摄像头权限被拒绝！请在浏览器设置中允许摄像头访问（地址栏左侧锁图标）');
                } else if (err.name === 'NotFoundError') {
                    alert('⚠️ 未检测到摄像头设备');
                } else {
                    alert('⚠️ 摄像头错误: ' + err.name);
                }
            });
        </script>
        """
        st.components.v1.html(js_code, height=0)

    col_monitor_btn, _ = st.sidebar.columns([1, 1])
    if not st.session_state["is_monitoring"]:
        if col_monitor_btn.button("▶️ 开始实时监控"):
            if st.session_state.get("camera") is not None:
                try: st.session_state["camera"].release()
                except: pass
                st.session_state["camera"] = None

            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            ret, test = cap.read()
            if not ret or test is None:
                cap.release()
                st.error("❌ 无法读取摄像头画面！\n\n可能原因：\n1. 浏览器未授权摄像头 → 请先点击「申请摄像头权限」\n2. 操作系统隐私设置禁止\n3. 摄像头被占用\n4. 无摄像头硬件")
            else:
                st.session_state["camera"] = cap
                st.session_state["is_monitoring"] = True
                st.session_state["smooth_cache"] = None
                st.session_state["frame_counter"] = 0
                st.success("✅ 摄像头已启动，实时分析中...")
                st.rerun()
    else:
        if col_monitor_btn.button("⏹️ 停止监控"):
            if st.session_state["camera"] is not None:
                st.session_state["camera"].release()
                st.session_state["camera"] = None
            st.session_state["is_monitoring"] = False
            st.rerun()

    # ---------- 实时监控显示 ----------
    if st.session_state["is_monitoring"]:
        cap = st.session_state["camera"]
        if cap is None or not cap.isOpened():
            st.error("摄像头连接已断开")
            st.session_state["is_monitoring"] = False
        else:
            ret, frame_bgr = cap.read()
            if not ret:
                st.error("读取摄像头画面失败")
            else:
                if st.session_state["mirror_state"]:
                    frame_bgr = cv2.flip(frame_bgr, 1)

                cnt = st.session_state["frame_counter"]
                do_inference = (cnt % st.session_state["skip_frames"] == 0)
                out_bgr, new_cache, face_count = process_frame(
                    frame_bgr, cls_model, vad_model,
                    st.session_state["smooth_cache"],
                    force_inference=do_inference
                )
                if do_inference:
                    st.session_state["smooth_cache"] = new_cache
                st.session_state["frame_counter"] = cnt + 1

                out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                # 显示时不压缩高度，设置固定宽度保持比例
                st.image(out_rgb, channels="RGB", width=800,
                         caption=f"实时情感识别（帧 #{cnt}，{'推理' if do_inference else '跳帧'}）")
                st.caption(f"检测到人脸数：{face_count}")

                buf = io.BytesIO()
                Image.fromarray(out_rgb).save(buf, format="PNG")
                st.download_button("📥 下载当前画面", buf.getvalue(),
                                   f"realtime_{time.strftime('%Y%m%d_%H%M%S')}.png")
        time.sleep(0.03)
        st.rerun()

    # ---------- 静态图片 / 视频上传 ----------
    if not st.session_state["is_monitoring"]:
        col_left, col_right = st.columns([1, 1.2])
        with col_left:
            st.subheader("📸 静态图片 / 视频分析")
            upload_img = st.file_uploader("上传静态图片", type=["jpg", "png", "jpeg"])
            upload_vid = st.file_uploader("上传视频（自动本地保存结果）", type=["mp4", "avi", "mov"])

        if upload_img:
            img_pil = Image.open(upload_img).convert("RGB")
            if st.session_state["mirror_state"]:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            out_bgr, _, face_count = process_frame(frame_bgr, cls_model, vad_model, force_inference=True)
            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            col_right.image(out_rgb, caption="图片情感分析结果", width=800)
            col_right.caption(f"检测到人脸数：{face_count}")
            buf = io.BytesIO()
            Image.fromarray(out_rgb).save(buf, format="PNG")
            col_right.download_button("下载分析图", buf.getvalue(), f"emo_{time.time():.0f}.png")

        if upload_vid:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(upload_vid.read())
                temp_video_path = tmp.name

            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            # 高度暂不设，等待第一帧生成后获取

            # 读取第一帧以确定输出尺寸
            ret, first_frame = cap.read()
            if not ret:
                st.error("视频读取失败")
                cap.release()
                os.unlink(temp_video_path)
            else:
                # 处理第一帧（镜像逻辑）
                if st.session_state["mirror_state"]:
                    first_frame = cv2.flip(first_frame, 1)
                out_bgr_first, smooth_cache, _ = process_frame(
                    first_frame, cls_model, vad_model, force_inference=True
                )
                out_h, out_w = out_bgr_first.shape[:2]

                # 重新定位视频读取头，准备从第一帧开始写入
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_video_path = os.path.join(VIDEO_OUTPUT_DIR, f"video_analysis_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))

                progress_bar = st.progress(0)
                status_text = st.empty()
                preview_holder = st.empty()

                # 写入第一帧
                out_writer.write(out_bgr_first)
                frame_idx = 1

                # 显示预览
                preview_rgb = cv2.cvtColor(out_bgr_first, cv2.COLOR_BGR2RGB)
                preview_holder.image(preview_rgb, channels="RGB", width=800, caption="视频分析预览")

                # 处理剩余帧
                ret, frame_bgr = cap.read()
                while ret:
                    if st.session_state["mirror_state"]:
                        frame_bgr = cv2.flip(frame_bgr, 1)

                    out_bgr, smooth_cache, _ = process_frame(
                        frame_bgr, cls_model, vad_model,
                        smooth_cache=smooth_cache, force_inference=True
                    )
                    out_writer.write(out_bgr)

                    progress = (frame_idx + 1) / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"处理中... {frame_idx+1}/{total_frames} 帧")

                    if frame_idx % 10 == 0:
                        preview_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
                        preview_holder.image(preview_rgb, channels="RGB", width=800,
                                             caption=f"预览（第 {frame_idx+1} 帧）")

                    frame_idx += 1
                    ret, frame_bgr = cap.read()

                cap.release()
                out_writer.release()
                os.unlink(temp_video_path)

                progress_bar.empty()
                status_text.empty()
                preview_holder.empty()

                st.success(f"✅ 视频分析完成！输出视频已保存至：{output_video_path}")
                st.info(f"总帧数：{frame_idx}，FPS：{fps:.2f}，时长：{frame_idx/fps:.1f} 秒")

                with open(output_video_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="📥 下载分析后的视频",
                    data=video_bytes,
                    file_name=f"video_analysis_{timestamp}.mp4",
                    mime="video/mp4"
                )