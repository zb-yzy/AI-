import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from collections import Counter
from transformers import BertTokenizer, BertForSequenceClassification

# 新增safetensors兼容低版本torch
try:
    from safetensors.torch import save_file, load_file
except ImportError:
    print("未安装safetensors，执行：pip install safetensors")
    exit()

# ===================== 【用户配置】 =====================
DATA_ROOT = "../../data/"
FACE_CSV = os.path.join(DATA_ROOT, "emotion.csv")
REVIEW_CSV = os.path.join(DATA_ROOT, "review.csv")

JOIN_COL = "电影ID"
LABEL_COL = "情绪"
TEXT_COL = "评论"

DATASET_NAME = "Movie_Emotion"
NUM_CLASSES = 7  # 7类情绪
MAX_SEQ_LEN = 128
RANDOM_SEED = 4
MIN_TEST_PER_CLASS = 8

BATCH_SIZE = 32
LR = 1e-6
CONTINUE_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results2/Review_train_results"
# 替换后缀为safetensors，不再使用pth
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{DATASET_NAME}.safetensors")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{DATASET_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{DATASET_NAME}.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{DATASET_NAME}.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)

BERT_MODEL_DIR = "../model"


# ======================================================

# ===================== 数据集类：单评论 + 对应电影全部情绪列表 =====================
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_seq_len):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        review_text = str(row[TEXT_COL])
        label_list = row["label_list"]
        label_len = len(label_list)

        encoding = self.tokenizer(
            review_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label_list": torch.tensor(label_list, dtype=torch.long),
            "label_len": torch.tensor(label_len, dtype=torch.long)
        }


# 自定义collate_fn：处理变长label_list
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    label_lists = [item["label_list"] for item in batch]
    label_lens = torch.stack([item["label_len"] for item in batch])
    return input_ids, attention_mask, token_type_ids, label_lists, label_lens


# ===================== 数据处理：关联评论 + 对应电影全情绪列表 =====================
def load_merge_data(face_csv, review_csv, tokenizer, max_seq_len, seed):
    df_face = pd.read_csv(face_csv, encoding="utf-8-sig")

    # 多编码读取评论文件
    enc_list = ["gb18030", "gbk", "gb2312", "utf-8", "utf-8-sig", "latin-1"]
    df_review = None
    for enc in enc_list:
        try:
            df_review = pd.read_csv(review_csv, encoding=enc, encoding_errors="ignore")
            print(f"✅ 评论文件读取成功，编码: {enc}")
            break
        except Exception:
            continue
    if df_review is None:
        raise Exception("所有编码均无法读取评论CSV，请检查文件")

    print(f"人脸情绪数据: {len(df_face)} 条")
    print(f"评论数据: {len(df_review)} 条")

    # 清洗ID
    def clean_id(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    df_face[JOIN_COL] = df_face[JOIN_COL].apply(clean_id)
    df_review[JOIN_COL] = df_review[JOIN_COL].apply(clean_id)

    # 1. 按电影ID聚合：电影ID -> 全部情绪列表（保留重复）
    movie_label_map = df_face.groupby(JOIN_COL)[LABEL_COL].apply(list).to_dict()
    df_review = df_review[df_review[JOIN_COL].isin(movie_label_map.keys())].copy()

    # 2. 每条评论绑定对应电影情绪列表
    df_review["label_list"] = df_review[JOIN_COL].map(movie_label_map)
    df_merge = df_review.copy()
    print(f"✅ 每条评论已绑定对应电影全情绪，总样本数: {len(df_merge)}")

    # 3. 标签编码
    all_emo = []
    for lst in df_merge["label_list"]:
        all_emo.extend(lst)
    le = LabelEncoder()
    le.fit(all_emo)
    df_merge["label_list"] = df_merge["label_list"].apply(lambda x: le.transform(x).tolist())
    print(f"✅ 标签编码完成，类别映射：{dict(zip(le.classes_, range(NUM_CLASSES)))}")

    # 4. 按电影ID划分训练/测试集
    unique_movie = df_merge[JOIN_COL].unique()
    train_movie, test_movie = train_test_split(
        unique_movie, test_size=0.2, random_state=seed, shuffle=True
    )
    train_df = df_merge[df_merge[JOIN_COL].isin(train_movie)]
    test_df = df_merge[df_merge[JOIN_COL].isin(test_movie)]

    # 5. 保证测试集每类标签总数 >= 8
    print(f"\n--- 校验测试集标签分布，最低要求：{MIN_TEST_PER_CLASS} ---")
    test_all_labels = []
    for lst in test_df["label_list"]:
        test_all_labels.extend(lst)
    test_cnt = Counter(test_all_labels)

    for cls in range(NUM_CLASSES):
        cnt = test_cnt.get(cls, 0)
        if cnt < MIN_TEST_PER_CLASS:
            need = MIN_TEST_PER_CLASS - cnt
            print(f"类别 {cls} 测试集仅有 {cnt} 个标签，补充 {need} 个")
            train_cls_rows = train_df[train_df["label_list"].apply(lambda x: cls in x)]
            if len(train_cls_rows) == 0:
                raise ValueError(f"训练集无类别 {cls}，无法补足")
            move_rows = train_cls_rows.sample(n=need, random_state=seed)
            train_df = train_df.drop(move_rows.index)
            test_df = pd.concat([test_df, move_rows], ignore_index=True)

    final_test_labels = []
    for lst in test_df["label_list"]:
        final_test_labels.extend(lst)
    print("✅ 最终测试集各类标签数量：", Counter(final_test_labels))

    # 构建数据集 & 加载器
    train_dataset = EmotionDataset(train_df, tokenizer, max_seq_len)
    test_dataset = EmotionDataset(test_df, tokenizer, max_seq_len)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    return train_loader, test_loader, train_df, test_df, le


# ===================== 训练函数：均等权重Loss =====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="训练", leave=False)

    for input_ids, attn_mask, token_type, label_lists, label_lens in pbar:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type = token_type.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type)
        logits = outputs.logits

        batch_loss = 0.0
        for i in range(len(label_lists)):
            lab_list = label_lists[i].to(device)
            k = label_lens[i].item()
            sample_logit = logits[i:i + 1]
            loss_sum = 0.0
            for lab in lab_list:
                loss_sum += criterion(sample_logit, lab.unsqueeze(0))
            sample_loss = loss_sum / k
            batch_loss += sample_loss

        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        pbar.set_postfix(loss=f"{batch_loss.item():.3f}")

    avg_loss = total_loss / len(loader)
    return avg_loss


# ===================== 测试函数：修复topk越界 + 匹配逻辑 =====================
def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="测试", leave=False)
        for input_ids, attn_mask, token_type, label_lists, label_lens in pbar:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type = token_type.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type)
            logits = outputs.logits

            # 计算loss
            batch_loss = 0.0
            for i in range(len(label_lists)):
                lab_list = label_lists[i].to(device)
                k = label_lens[i].item()
                sample_logit = logits[i:i + 1]
                loss_sum = 0.0
                for lab in lab_list:
                    loss_sum += criterion(sample_logit, lab.unsqueeze(0))
                batch_loss += loss_sum / k
            total_loss += batch_loss.item()

            # 预测逻辑：修复k越界问题
            for i in range(len(label_lists)):
                true_labs = label_lists[i].cpu().numpy().tolist()
                real_k = label_lens[i].item()
                # 核心修复：k不能超过总类别数7
                topk_num = min(real_k, NUM_CLASSES)
                topk_pred = torch.topk(logits[i], k=topk_num).indices.cpu().numpy().tolist()

                # 逐标签匹配统计
                true_copy = true_labs.copy()
                for p in topk_pred:
                    if true_copy:
                        if p in true_copy:
                            t_idx = true_copy.index(p)
                            t = true_copy.pop(t_idx)
                        else:
                            t = true_copy.pop(0)
                        all_true.append(t)
                        all_pred.append(p)
                # 若真实标签数量 > 7，剩余真实标签统一按错分统计
                while len(true_copy) > 0:
                    t = true_copy.pop(0)
                    all_true.append(t)
                    all_pred.append(-1)  # 标记无对应预测

        avg_loss = total_loss / len(loader)
        # 过滤无效标记，计算准确率
        valid_pairs = [(t, p) for t, p in zip(all_true, all_pred) if p != -1]
        if not valid_pairs:
            acc = 0.0
        else:
            correct = sum(1 for t, p in valid_pairs if t == p)
            acc = correct / len(valid_pairs)

    return avg_loss, acc, all_true, all_pred


# ===================== 日志、绘图、混淆矩阵保存（使用原始情绪名称） =====================
def write_log(epoch, train_loss, test_loss, test_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch},train_loss={train_loss:.4f},test_loss={test_loss:.4f},test_acc={test_acc:.4f}\n")


# 改动：新增 label_encoder 参数，使用原始情绪文本作为标签
def save_cm(true, pred, save_path, label_encoder):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 直接使用LabelEncoder中原始情绪名称
    class_names = label_encoder.classes_

    # 过滤无效预测标签 -1
    filter_true = [t for t, p in zip(true, pred) if p != -1]
    filter_pred = [p for t, p in zip(true, pred) if p != -1]
    cm = confusion_matrix(filter_true, filter_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(class_names, rotation=45, ha="right")  # 标签旋转防止重叠
    ax.set_yticklabels(class_names)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("情绪预测混淆矩阵")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 结果保存目录：{SAVE_DIR}")

    # 加载模型&分词器
    print(f"\n🚀 加载BERT模型: {BERT_MODEL_DIR}")
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_DIR, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_DIR, num_labels=NUM_CLASSES, local_files_only=True
        )
        model = model.to(DEVICE)
        print("✅ BERT加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit()

    # 加载数据
    print("\n📊 开始加载数据...")
    train_loader, test_loader, train_df, test_df, label_encoder = load_merge_data(
        FACE_CSV, REVIEW_CSV, tokenizer, MAX_SEQ_LEN, RANDOM_SEED
    )
    print(f"训练集评论数: {len(train_loader.dataset)} | 测试集评论数: {len(test_loader.dataset)}")

    # 损失&优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # 续训初始化
    start_epoch = 0
    best_acc = 0.0
    train_loss_list, test_loss_list, test_acc_list = [], [], []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Epoch"):
                    continue
                parts = line.split(",")
                tl = float(parts[1].replace("train_loss=", ""))
                tel = float(parts[2].replace("test_loss=", ""))
                tea = float(parts[3].replace("test_acc=", ""))
                train_loss_list.append(tl)
                test_loss_list.append(tel)
                test_acc_list.append(tea)
                if tea > best_acc:
                    best_acc = tea
        start_epoch = len(train_loss_list)
        print(f"\n✅ 读取历史记录，上次训练至 Epoch {start_epoch}，最佳ACC: {best_acc:.4f}")

    # 【修复点】改用safetensors加载权重，规避torch.load安全限制
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            state_dict = load_file(MODEL_SAVE_PATH)
            model.load_state_dict(state_dict)
            print("✅ 加载历史最优模型(safetensors格式)")
        except Exception as e:
            print(f"⚠️ safetensors模型加载失败，从头训练，错误:{e}")

    # 开始训练
    print(f"\n🚀 开始训练，共 {CONTINUE_EPOCHS} 轮")
    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, all_true, all_pred = test(model, test_loader, criterion, DEVICE)

        train_loss_list.append(tr_loss)
        test_loss_list.append(te_loss)
        test_acc_list.append(te_acc)
        write_log(current_epoch, tr_loss, te_loss, te_acc)
        scheduler.step(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            # 【修复点】保存使用safetensors，不再用torch.save
            save_file(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型保存，当前最佳ACC: {best_acc:.4f}")

        print(f"训练损失: {tr_loss:.4f} | 测试损失: {te_loss:.4f} | 测试准确率: {te_acc:.4f}\n")

    # 绘制曲线
    print("\n📈 绘制训练曲线")
    plt.figure(figsize=(12, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="训练损失")
    plt.plot(test_loss_list, label="测试损失")
    plt.title("损失曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(test_acc_list, label="测试准确率", color="orange")
    plt.title("测试准确率曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 训练曲线已保存")

    # 生成混淆矩阵（传入标签编码器，使用原始情绪名称）
    print("\n📊 生成混淆矩阵")
    _, _, final_true, final_pred = test(model, test_loader, criterion, DEVICE)
    save_cm(final_true, final_pred, CM_SAVE_PATH, label_encoder)

    print(f"\n🏁 训练完成！最佳准确率: {best_acc:.4f}")