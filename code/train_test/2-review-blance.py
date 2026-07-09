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
import seaborn as sns
import random

# ===================== 【全局随机种子 + 配置】 =====================
SEED = 4
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_ROOT = "../../data/"
FACE_CSV = os.path.join(DATA_ROOT, "emotion.csv")
REVIEW_CSV = os.path.join(DATA_ROOT, "review.csv")

JOIN_COL = "电影ID"
LABEL_COL = "情绪"
TEXT_COL = "评论"

DATASET_NAME = "../../results/Movie_Emotion"
NUM_CLASSES = 7          # 7类情绪
MAX_SEQ_LEN = 128
RANDOM_SEED = 4
MIN_TEST_PER_CLASS = 8

BATCH_SIZE = 32
LR = 1e-6
CONTINUE_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "../../results/results2/Review_blance_results"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, f"best_model_{DATASET_NAME}.pth")
LOG_PATH = os.path.join(SAVE_DIR, f"train_log_{DATASET_NAME}.txt")
CURVE_SAVE_PATH = os.path.join(SAVE_DIR, f"train_curve_{DATASET_NAME}.png")
CM_SAVE_PATH = os.path.join(SAVE_DIR, f"confusion_matrix_{DATASET_NAME}.png")
PRED_SAVE_DIR = os.path.join(SAVE_DIR, "test_predictions")
os.makedirs(PRED_SAVE_DIR, exist_ok=True)

BERT_MODEL_DIR = "../model"
# ======================================================

# ===================== 数据集类 =====================
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

# 自定义collate_fn
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    label_lists = [item["label_list"] for item in batch]
    label_lens = torch.stack([item["label_len"] for item in batch])
    return input_ids, attention_mask, token_type_ids, label_lists, label_lens

# ===================== 数据处理 =====================
def load_merge_data(face_csv, review_csv, tokenizer, max_seq_len, seed):
    df_face = pd.read_csv(face_csv, encoding="utf-8-sig")

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

    def clean_id(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s
    df_face[JOIN_COL] = df_face[JOIN_COL].apply(clean_id)
    df_review[JOIN_COL] = df_review[JOIN_COL].apply(clean_id)

    movie_label_map = df_face.groupby(JOIN_COL)[LABEL_COL].apply(list).to_dict()
    df_review = df_review[df_review[JOIN_COL].isin(movie_label_map.keys())].copy()

    df_review["label_list"] = df_review[JOIN_COL].map(movie_label_map)
    df_merge = df_review.copy()
    print(f"✅ 每条评论已绑定对应电影全情绪，总样本数: {len(df_merge)}")

    all_emo = []
    for lst in df_merge["label_list"]:
        all_emo.extend(lst)
    le = LabelEncoder()
    le.fit(all_emo)
    df_merge["label_list"] = df_merge["label_list"].apply(lambda x: le.transform(x).tolist())
    print(f"✅ 标签编码完成，类别映射：{dict(zip(le.classes_, range(NUM_CLASSES)))}")

    unique_movie = df_merge[JOIN_COL].unique()
    train_movie, test_movie = train_test_split(
        unique_movie, test_size=0.2, random_state=seed, shuffle=True
    )
    train_df = df_merge[df_merge[JOIN_COL].isin(train_movie)]
    test_df = df_merge[df_merge[JOIN_COL].isin(test_movie)]

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

    # 收集训练集所有标签，用于计算类别权重
    train_all_labels = []
    for lst in train_df["label_list"]:
        train_all_labels.extend(lst)

    return train_loader, test_loader, train_df, test_df, le, train_all_labels

# ===================== 训练函数 =====================
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
            sample_logit = logits[i:i+1]
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

# ===================== 测试函数（新增平衡准确率） =====================
def test(model, loader, criterion, device, num_classes=NUM_CLASSES):
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

            batch_loss = 0.0
            for i in range(len(label_lists)):
                lab_list = label_lists[i].to(device)
                k = label_lens[i].item()
                sample_logit = logits[i:i+1]
                loss_sum = 0.0
                for lab in lab_list:
                    loss_sum += criterion(sample_logit, lab.unsqueeze(0))
                batch_loss += loss_sum / k
            total_loss += batch_loss.item()

            for i in range(len(label_lists)):
                true_labs = label_lists[i].cpu().numpy().tolist()
                real_k = label_lens[i].item()
                topk_num = min(real_k, NUM_CLASSES)
                topk_pred = torch.topk(logits[i], k=topk_num).indices.cpu().numpy().tolist()

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
                while len(true_copy) > 0:
                    t = true_copy.pop(0)
                    all_true.append(t)
                    all_pred.append(-1)

        avg_loss = total_loss / len(loader)
        # 过滤无效标签
        valid_true = [t for t,p in zip(all_true, all_pred) if p != -1]
        valid_pred = [p for t,p in zip(all_true, all_pred) if p != -1]

        # 总体准确率
        overall_acc = 0.0
        if valid_true:
            correct = sum(1 for t,p in zip(valid_true, valid_pred) if t == p)
            overall_acc = correct / len(valid_true)

        # 逐类别准确率 + 平衡准确率
        per_class_acc = []
        for c in range(num_classes):
            mask = np.array(valid_true) == c
            if mask.sum() == 0:
                per_class_acc.append(0.0)
            else:
                class_correct = np.sum((np.array(valid_pred) == c) & mask)
                per_class_acc.append(class_correct / mask.sum())
        balanced_acc = np.mean(per_class_acc)

    return avg_loss, overall_acc, balanced_acc, all_true, all_pred, valid_true, valid_pred

# ===================== 日志、绘图、混淆矩阵（使用原始情绪名称） =====================
def write_log(epoch, train_loss, test_loss, test_acc, balanced_acc):
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(
            f"Epoch {epoch},train_loss={train_loss:.4f},test_loss={test_loss:.4f},"
            f"test_acc={test_acc:.4f},balanced_acc={balanced_acc:.4f}\n"
        )

# 传入标签编码器，使用CSV原始情绪名
def save_cm(true, pred, save_path, label_encoder):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    class_names = label_encoder.classes_
    cm = confusion_matrix(true, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("情绪预测混淆矩阵(归一化)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")

# ===================== 主程序 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"📁 结果保存目录：{SAVE_DIR}")

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

    print("\n📊 开始加载数据...")
    train_loader, test_loader, train_df, test_df, label_encoder, train_all_labels = load_merge_data(
        FACE_CSV, REVIEW_CSV, tokenizer, MAX_SEQ_LEN, RANDOM_SEED
    )
    print(f"训练集评论数: {len(train_loader.dataset)} | 测试集评论数: {len(test_loader.dataset)}")

    # ========== 计算类别权重，解决数据不平衡 ==========
    train_label_arr = np.array(train_all_labels)
    class_counts = np.bincount(train_label_arr, minlength=NUM_CLASSES)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"📊 类别平衡权重: {class_weights.cpu().numpy()}")

    # 使用加权损失函数
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # 续训初始化
    start_epoch = 0
    best_balanced_acc = 0.0
    train_loss_list, test_loss_list, test_acc_list, balanced_acc_list = [], [], [], []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Epoch"):
                    continue
                parts = line.split(",")
                tl = float(parts[1].replace("train_loss=",""))
                tel = float(parts[2].replace("test_loss=",""))
                tea = float(parts[3].replace("test_acc=",""))
                ba = float(parts[4].replace("balanced_acc=",""))
                train_loss_list.append(tl)
                test_loss_list.append(tel)
                test_acc_list.append(tea)
                balanced_acc_list.append(ba)
                if ba > best_balanced_acc:
                    best_balanced_acc = ba
        start_epoch = len(train_loss_list)
        print(f"\n✅ 读取历史记录，上次训练至 Epoch {start_epoch}，最佳平衡ACC: {best_balanced_acc:.4f}")

    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
            print("✅ 加载历史最优模型")
        except:
            print("⚠️ 模型加载失败，从头训练")

    print(f"\n🚀 开始训练，共 {CONTINUE_EPOCHS} 轮")
    for i in range(CONTINUE_EPOCHS):
        current_epoch = start_epoch + i + 1
        print(f"======== Epoch {current_epoch} ========")

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        te_loss, te_acc, bal_acc, all_true, all_pred, valid_true, valid_pred = test(
            model, test_loader, criterion, DEVICE
        )

        train_loss_list.append(tr_loss)
        test_loss_list.append(te_loss)
        test_acc_list.append(te_acc)
        balanced_acc_list.append(bal_acc)
        write_log(current_epoch, tr_loss, te_loss, te_acc, bal_acc)
        scheduler.step(bal_acc)

        # 以平衡准确率保存最优模型
        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ 新最优模型保存，当前最佳平衡ACC: {best_balanced_acc:.4f}")

        print(f"训练损失: {tr_loss:.4f} | 测试损失: {te_loss:.4f}")
        print(f"测试总体准确率: {te_acc:.4f} | 测试平衡准确率: {bal_acc:.4f}\n")

    # ========== 绘制多曲线 ==========
    print("\n📈 绘制训练曲线")
    plt.figure(figsize=(15, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(1, 3, 1)
    plt.plot(train_loss_list, label="训练损失")
    plt.plot(test_loss_list, label="测试损失")
    plt.title("损失变化曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(test_acc_list, label="测试总体准确率", color="orange")
    plt.title("测试总体准确率曲线")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(balanced_acc_list, label="测试平衡准确率", color="green")
    plt.title("测试平衡准确率曲线(各类别均值)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVE_SAVE_PATH, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"✅ 训练曲线已保存")

    # ========== 生成归一化混淆矩阵 + 输出每类准确率 ==========
    print("\n📊 生成混淆矩阵 & 类别准确率")
    _, _, _, final_true, final_pred, valid_true, valid_pred = test(model, test_loader, criterion, DEVICE)
    # 传入标签编码器，使用原始情绪名称
    save_cm(valid_true, valid_pred, CM_SAVE_PATH, label_encoder)

    # 输出每一类准确率（替换为原始情绪名）
    per_class_acc = []
    emotion_names = label_encoder.classes_
    for c in range(NUM_CLASSES):
        mask = np.array(valid_true) == c
        if mask.sum() == 0:
            per_class_acc.append(np.nan)
        else:
            class_correct = np.sum((np.array(valid_pred) == c) & mask)
            per_class_acc.append(class_correct / mask.sum())

    print("\n📈 各类别准确率明细：")
    for idx, (name, acc) in enumerate(zip(emotion_names, per_class_acc)):
        if np.isnan(acc):
            print(f"{name} : 无测试样本")
        else:
            print(f"{name} : {acc:.4f}")

    print(f"\n🏁 训练完成！最佳平衡准确率: {best_balanced_acc:.4f}")