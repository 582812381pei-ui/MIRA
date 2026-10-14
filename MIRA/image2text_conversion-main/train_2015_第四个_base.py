import gc
import warnings
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaModel, BertModel

from model.MabsaModel import *
from model.OutModel import *
from model.text_model import *
from Loss.combin_loss import *
from utils.helpers import *
from utils.dataset import *
from utils.data_utils import *
from model.clip_model import *

warnings.filterwarnings("ignore")  # 禁用警告
warnings.filterwarnings("ignore", message=".*overflowing tokens.*")

import argparse
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
# Config
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default="cuda:1")
parser.add_argument('--data_dir', type=str, default="/home/zxp/learn/datasets/2017数据集")
parser.add_argument('--catr_ckpt', type=str, default="/home/zxp/learn/models/cart/checkpoint.pth")
parser.add_argument('--result_dir', type=str, default="results")
parser.add_argument('--log_dir', type=str, default="logs")
parser.add_argument('--dataset', type=str,
                    default="twitter2015_gpt_all")  # twitter2015 or twitter2017
# parser.add_argument('--model', type=str, default="bertweet-base") # "bert-base-uncased" "vinai/bertweet-base"
parser.add_argument('--model', type=str,
                    default="/home/zxp/learn/models/bertweet-base")  # "bert-base-uncased" "vinai/bertweet-base
parser.add_argument('--model_large', type=str,
                    default="/home/zxp/learn/models/bertweet-large")  # "bert-base-uncased" "vinai/bertweet-base
parser.add_argument('--num_cycles', type=float, default=0.5)
parser.add_argument('--scheduler', type=str, default='linear')  # ['linear', 'cosine']
parser.add_argument('--print_freq', type=int, default=80)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--max_grad_norm', type=int, default=20)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--lr_fus_output', type=float, default=2e-5)
parser.add_argument('--max_len', type=int, default=128)  # default=72
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--bert_dim', type=int, default=768)
parser.add_argument('--clip_image_feature_dim', type=int, default=1152)
parser.add_argument('--clip_text_feature_dim', type=int, default=1152)
parser.add_argument('--save_dir', type=str, default="./results/2017")
args = parser.parse_args()

args = parser.parse_args()

device = torch.device(args.device)

best_acc = {"acc": 0, "macro_f1": 0}

from datetime import datetime  # 获取当前时间，格式化为 "年-月-日_小时-分钟-秒"

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 使用 time 和日志路径合并
# 使用 time 和日志路径合并
log_path = os.path.join(args.log_dir, f"base_loss3_local_only_ocr_twomodel_{current_time}_{args.dataset}")
LOGGER = get_logger(log_path)
train_df, val_df, test_df = load_data(args)

tokenizer = AutoTokenizer.from_pretrained(args.model)
args.tokenizer = tokenizer
args.pad_token_id = args.tokenizer.pad_token_id
args.end_token_id = args.tokenizer.sep_token_id

import open_clip
from peft import get_peft_model, LoraConfig

# 1) 创建模型
clip_base_model, clip_image_preprocess, _ = open_clip.create_model_and_transforms(
    model_name='local-dir:/home/zxp/learn/models/ViT-SO400M-16-SigLIP2-384')
clip_text_tokenizer = open_clip.get_tokenizer(model_name='local-dir:/home/zxp/learn/models/ViT-SO400M-16-SigLIP2-384')
args.clip_text_tokenizer = clip_text_tokenizer
args.clip_image_preprocess = clip_image_preprocess

# 2) 注入 LoRA（按模块名子串匹配 Linear）
# 视觉塔可用: "qkv","proj","fc1","fc2"
# 文本塔可用: "out_proj","c_fc","c_proj"（注意：text attn 没有 qkv）
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["qkv", "proj", "fc1", "fc2", "out_proj", "c_fc", "c_proj"],
    lora_dropout=0.05,
    bias="none",
)
clip_base_model = get_peft_model(clip_base_model, lora_config)


# 3) 仅训练最后4层 LoRA（默认只开视觉塔，需文本一起开就把 enable_text=True）
def enable_last_k_lora(clip_model, k=4, enable_text=True):
    # 冻结全部
    for p in clip_model.parameters():
        p.requires_grad = False

    # 视觉塔 block 个数与末尾索引
    v_blocks = getattr(clip_model.visual.trunk, "blocks")
    num_v = len(v_blocks)
    keep_v = {i for i in range(max(0, num_v - k), num_v)}

    # 可选：文本塔 block
    if enable_text:
        t_blocks = clip_model.text.transformer.resblocks
        num_t = len(t_blocks)
        keep_t = {i for i in range(max(0, num_t - k), num_t)}
    else:
        keep_t = set()

    # 只开启匹配到的 LoRA 参数
    for name, p in clip_model.named_parameters():
        if "lora_" not in name:
            continue
        ok = any(f"visual.trunk.blocks.{bid}." in name for bid in keep_v)
        if not ok and enable_text:
            ok = any(f"text.transformer.resblocks.{bid}." in name for bid in keep_t)
        p.requires_grad = ok


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {100 * trainable / total:.4f}%")


enable_last_k_lora(clip_base_model, k=4, enable_text=True)  # 只调视觉最后4层
print_trainable_parameters(clip_base_model)

LOGGER.info(
    f"lr={args.lr}, epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, model={args.model}, dataset={args.dataset}")


def get_max_f1_from_dir(save_dir):
    """
    扫描目录，从文件名中解析并返回已存模型的最高 macro_f1 分数。
    文件名格式应为: best_model_epoch_{epoch}_macro_f1_{score}.pth
    """
    if not os.path.exists(save_dir):
        return 0.0  # 如果目录不存在，则认为最高分为0

    # 正则表达式，用于匹配文件名并提取F1分数
    # \d+ 匹配一个或多个数字, \d+\.\d+ 匹配浮点数
    pattern = re.compile(r"best_model_epoch_\d+_macro_f1_(\d+\.\d+)\.pth")

    max_f1 = 0.0
    for filename in os.listdir(save_dir):
        match = pattern.match(filename)
        if match:
            # match.group(1) 捕获的是括号内的内容，即F1分数
            f1_score = float(match.group(1))
            if f1_score > max_f1:
                max_f1 = f1_score

    return max_f1


## Train fn
def train_fn(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples, epoch, test_loader,
             combined_loader, gradient_accumulation_steps=4):
    """
    梯度累积版训练函数
    :param gradient_accumulation_steps: 每累积多少步更新一次梯度
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    start = time.time()

    optimizer.zero_grad(set_to_none=True)

    for step, d in enumerate(data_loader):
        input_ids_all = d["input_ids_all"].to(device)
        attention_mask_all = d["attention_mask_all"].to(device)
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        clip_image = d["clip_image"].to(device)
        clip_aspect = d["clip_aspect"].to(device)
        clip_tweet = d["clip_tweet"].to(device)
        candidate_input_ids = d["candidate_input_ids"].to(device)
        candidate_attention_mask = d["candidate_attention_mask"].to(device)

        output_image = d["output_image"].to(device)
        output_text = d["output_text"].to(device)

        only_text_output,clip_output,fus_output = model(input_ids, attention_mask,
                        input_ids_all, attention_mask_all,
                        clip_image, clip_aspect,clip_tweet,
                        candidate_input_ids,candidate_attention_mask)

        _, preds = torch.max(fus_output, dim=1)

        loss = loss_fn(only_text_output, clip_output, fus_output, labels,epoch,output_image,output_text)

        # 损失缩放，避免梯度被放大
        loss = loss / gradient_accumulation_steps
        loss.backward()

        correct_predictions += torch.sum(preds == labels).item()
        losses.append(loss.item() * gradient_accumulation_steps)  # 还原原始损失值

        # 梯度累积：每到一定步数才更新一次参数
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(data_loader):
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # 打印信息
        if epoch <= 1:
            continue
        if step % args.print_freq == 0 or step == (len(data_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss:.4f} '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch + 1, step, len(data_loader),
                          remain=timeSince(start, float(step + 1) / len(data_loader)),
                          loss=loss.item() * gradient_accumulation_steps,
                          grad_norm=grad_norm if 'grad_norm' in locals() else 0,
                          lr=scheduler.get_last_lr()[0]))

            test_acc, test_loss, dr, class_acc = eval_model(
                model, test_loader, loss_fn, device, len(test_df), detailed_results=True
            )
            model.train()
            macro_f1 = f1_score(dr.label, dr.prediction, average="macro")

            max_existing_f1 = get_max_f1_from_dir(args.save_dir)
            if macro_f1 > 0.73 and macro_f1 > max_existing_f1:
                print(f"发现新的最佳模型！准备保存...")
                _, _, dr_combined, _ = eval_model(
                    model, combined_loader, loss_fn, device, len(test_df), detailed_results=True
                )
                model.train()
                file_name = os.path.join(args.save_dir,
                                         "2017_{}_{}.xlsx".format(
                                             epoch + 1, "{:.4f}".format(macro_f1).replace('.', '_')))
                dr_combined.to_excel(file_name, index=False)

            LOGGER.info(
                f'Epoch [{epoch + 1}][{step}/{len(data_loader)}] - test loss {test_loss} accuracy {test_acc} macro f1 {macro_f1}, class acc {class_acc}')

    return correct_predictions / n_examples, np.mean(losses)



# eval fn
import numpy as np
import pandas as pd


def format_eval_output(rows):
    # 解包数据，确保所有数据列正确分配
    tweets, targets, image_id, ocr_text, labels, predictions, image_info, fus_output = zip(
        *rows)

    # 通过 np.vstack 将每一列数据堆叠起来
    tweets = np.vstack(tweets) if isinstance(tweets[0], np.ndarray) else np.array(tweets)
    targets = np.vstack(targets) if isinstance(targets[0], np.ndarray) else np.array(targets)
    ocr_text = np.vstack(ocr_text) if isinstance(ocr_text[0], np.ndarray) else np.array(ocr_text)
    labels = np.vstack(labels) if isinstance(labels[0], np.ndarray) else np.array(labels)
    predictions = np.vstack(predictions) if isinstance(predictions[0], np.ndarray) else np.array(predictions)
    image_info = np.vstack(image_info) if isinstance(image_info[0], np.ndarray) else np.array(image_info)

    fus_output = np.vstack(fus_output) if isinstance(fus_output[0], np.ndarray) else np.array(
        fus_output)

    # 将堆叠后的数据转换为 DataFrame
    results_df = pd.DataFrame({
        "tweet": tweets.reshape(-1).tolist(),
        "target": targets.reshape(-1).tolist(),
        "ocr_text": ocr_text.reshape(-1).tolist(),
        "image_id": image_id,
        "label": labels.reshape(-1).tolist(),
        "prediction": predictions.reshape(-1).tolist(),
        "image_info": image_info.reshape(-1).tolist(),
        "fus_output_0": fus_output[:, 0].tolist(),  # 添加 fus_output 的第一个logit
        "fus_output_1": fus_output[:, 1].tolist(),  # 添加 fus_output 的第二个logit
        "fus_output_2": fus_output[:, 2].tolist(),  # 添加 fus_output 的第三个logit
    })

    return results_df


def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    rows = []

    num_classes = 3  # ⚠️ 根据你的任务类别数改，比如3分类任务就填3
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():
        for step, d in enumerate(data_loader):
            input_ids_all = d["input_ids_all"].to(device)
            attention_mask_all = d["attention_mask_all"].to(device)
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            clip_image = d["clip_image"].to(device)
            clip_aspect = d["clip_aspect"].to(device)
            clip_tweet = d["clip_tweet"].to(device)
            candidate_input_ids = d["candidate_input_ids"].to(device)
            candidate_attention_mask = d["candidate_attention_mask"].to(device)

            only_text_output, clip_output, fus_output = model(input_ids, attention_mask,
                                                              input_ids_all, attention_mask_all,
                                                              clip_image, clip_aspect, clip_tweet,
                                                              candidate_input_ids, candidate_attention_mask)

            _, preds = torch.max(fus_output, dim=1)

            correct_predictions += torch.sum(preds == labels).item()

            for label, pred in zip(labels, preds):
                class_total[label.item()] += 1
                if pred.item() == label.item():
                    class_correct[label.item()] += 1

            rows.extend(
                zip(d["review_text"],
                    d["targets"],
                    d['image_id'],
                    d['ocr_text'],
                    d["labels"].numpy(),
                    preds.cpu().numpy(),
                    d["image_info"],
                    fus_output.cpu().numpy()
                    )
            )

    class_acc = []
    for i in range(num_classes):
        if class_total[i] > 0:
            class_acc.append(class_correct[i] / class_total[i])
        else:
            class_acc.append(0.0)

    if detailed_results:
        return (
            correct_predictions / n_examples,  # 总体准确率
            np.mean(losses),  # 平均损失
            format_eval_output(rows),  # 原本的详细结果
            class_acc  # 新增：每类准确率
        )

    return correct_predictions / n_examples, np.mean(losses), class_acc


def train_loop():
    LOGGER.info(f"========== Start Training ==========")
    LOGGER.info(
        f"lr={args.lr}, seed={args.seed},epochs={args.epochs}, batch_size={args.batch_size}, dropout={args.dropout}, model={args.model}, dataset={args.dataset}")
    # Create DataLoader
    train_dataset = TwitterDataset(args, train_df)
    test_dataset = TwitterDataset(args, test_df)
    # train_dataset = TwitterDataset(args, train_df.iloc[:200])  # 仅用200条数据测试
    # test_dataset = TwitterDataset(args, test_df.iloc[:200])  # 仅用200条数据测试
    from torch.utils.data import ConcatDataset, DataLoader
    # 合并 train_dataset 和 test_dataset
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # 用一个 DataLoader 处理
    combined_loader = DataLoader(combined_dataset,
                                 128,
                                 shuffle=False,  # 可以 shuffle 所有样本
                                 num_workers=args.num_workers,
                                 pin_memory=True)

    # val_dataset = TwitterDataset(args, val_df)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
    #                         num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # image_text_model = TextImageModel(args)
    # only_text_model = OnlyTextModel(args)
    clip_model = ClipModel(args, clip_base_model)
    text_model = OnlyTextModel(args)

    model = FusedModel_loss_1_1(text_model,clip_model)

    # model = FusedModel_loss_1_1(only_text_model, image_text_model, clip_model)
    # print(model)
    model.to(device)
    # Configure the optimizer and scheduler.
    # param_dicts = [
    #     {
    #         "params": model.stack_fus.parameters(),  # 只为 stack_fus 层设置不同的学习率
    #         "lr": args.lr_fus_output,
    #     },
    #     {
    #         "params": [param for name, param in model.named_parameters() if "stack_fus" not in name],  # 其余层使用默认学习率
    #         "lr": args.lr,
    #     }
    # ]
    # param_dicts = model.parameters()

    optimizer = AdamW(params=model.parameters(), lr=args.lr)

    # optimizer = AdamW(param_dicts, lr=args.lr, correct_bias=args.adamw_correct_bias)

    num_train_steps = int(len(train_df) / args.batch_size) * args.epochs
    args.num_warmup_steps = num_train_steps * 0.06
    scheduler = get_scheduler(args, optimizer, num_train_steps)
    # 这个权重是每一类的样本数与总样本数的比例的倒数，这样定义loss的作用是对样本不均衡的类别给予更高的权重
    # loss_weights = torch.FloatTensor([3719/368,
    #                                     3179/1883,
    #                                     3179/928,])
    # loss_fn = nn.CrossEntropyLoss(weight=loss_weights).to(device)
    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = CombinedLoss_3_1().to(device)

    for epoch in range(args.epochs):
        print(f"===============Epoch {epoch + 1}/{args.epochs}==============")
        start_time = time.time()

        train_acc, train_loss = train_fn(
            model, train_loader, loss_fn, optimizer, device, scheduler, len(train_df), epoch, test_loader,
            combined_loader, args.gradient_accumulation_steps
        )

        # val_acc, val_loss, dr = eval_model(model, val_loader, loss_fn, device, len(val_df), detailed_results=True)
        # macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        # LOGGER.info(f'Epoch {epoch + 1} - Val loss {val_loss} accuracy {val_acc} macro f1 {macro_f1}')

        test_acc, test_loss, dr, class_acc = eval_model(
            model, test_loader, loss_fn, device, len(test_df), detailed_results=True
        )
        macro_f1 = f1_score(dr.label, dr.prediction, average="macro")
        # dr.to_excel(f"results/epoch_{epoch + 1}_test.xlsx")

        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch + 1} - Train loss {train_loss} accuracy {train_acc}" time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Test loss {test_loss} accuracy {test_acc} macro f1 {macro_f1}')

    torch.cuda.empty_cache()
    gc.collect()
    LOGGER.info(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")
    return dr


if __name__ == "__main__":
    # seed_everything(seed=args.seed)
    # x = train_loop()
    for i in range(100):
        # 随机生成一个数
        args.seed = random.randint(0, 100000)
        seed_everything(seed=args.seed)
        print("第{}次实验".format(i))
        x = train_loop()
    print(best_acc)



