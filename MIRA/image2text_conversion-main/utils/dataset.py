import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import warnings
import torch
import ast

from transformers import AutoTokenizer

warnings.filterwarnings("ignore")


class TwitterDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len
        self.tokenizer_large = AutoTokenizer.from_pretrained(self.cfg.model_large)

        # ====== 将所有数据一次性预处理好缓存到内存 ======
        self.samples = []
        print(f"🔄 正在预处理 {len(df)} 条样本...")

        for idx in range(len(df)):
            row = df.iloc[idx]
            sample = self._process_one(row)
            self.samples.append(sample)

        print(f"✅ 数据预处理完成，共缓存 {len(self.samples)} 条样本。")

    def _process_one(self, row):
        tweet = str(row["tweet_content"])
        ocr_text = str(row["ocr_text"])
        image_id = str(row["image_id"])
        target = str(row["target"])
        image_info = str(row["image_info"])
        label = int(row["sentiment"])
        candidate_aspects = eval(row["nouns"])
        output_image = ast.literal_eval(row["output_image"])
        output_text = ast.literal_eval(row["output_text"])
        output_image = torch.tensor(output_image, dtype=torch.float32)
        output_text = torch.tensor(output_text, dtype=torch.float32)
        # ---------- 文本拼接 ----------
        if ocr_text != "no text":
            tweet_all = tweet + " " + ocr_text
        else:
            tweet_all = tweet + ". " + image_info
        if ocr_text != "no text":
            tweet = tweet + " " + ocr_text

        # ---------- 加载图像 ----------
        if "2017" in self.cfg.dataset:
            image_dir = os.path.join(
                "/home/zxp/learn/datasets/IJCAI2019_data/twitter2017_images/", image_id
            )
        else:
            image_dir = os.path.join(
                "/home/zxp/learn/datasets/IJCAI2019_data/twitter2015_images/", image_id
            )
        image = Image.open(image_dir).convert("RGB")
        image_tensor = self.cfg.clip_image_preprocess(image)

        # ---------- Clip 模型文本 ----------
        tweet_target = tweet.replace("$T$", target)
        clip_target = self.cfg.clip_text_tokenizer([target])
        clip_tweet = self.cfg.clip_text_tokenizer([tweet_target])

        # ---------- Bert 文本 ----------
        encoding = self.tokenizer_large.encode_plus(
            tweet,
            text_pair=target,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        encoding_all = self.tokenizer_large.encode_plus(
            tweet_all,
            text_pair=target,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation="longest_first",
        )

        # ---------- 候选方面词 ----------
        if len(candidate_aspects) > 3:
            candidate_aspects = random.sample(candidate_aspects, 3)

        candidate_input_ids = []
        candidate_attention_mask = []
        for asp in candidate_aspects:
            candidate_encoding = self.tokenizer_large.encode_plus(
                tweet_target.replace(asp, "$T$"),
                text_pair=asp,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation="longest_first",
            )
            candidate_input_ids.append(candidate_encoding["input_ids"].flatten())
            candidate_attention_mask.append(candidate_encoding["attention_mask"].flatten())

        # 如果不足 3 个，填充
        while len(candidate_input_ids) < 3:
            candidate_input_ids.append(encoding["input_ids"].flatten())
            candidate_attention_mask.append(encoding["attention_mask"].flatten())

        return {
            "review_text": tweet,
            "targets": target,
            "image_id": image_id,
            "ocr_text": ocr_text,
            "image_info": image_info,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "input_ids_all": encoding_all["input_ids"].flatten(),
            "attention_mask_all": encoding_all["attention_mask"].flatten(),
            "clip_image": image_tensor,
            "clip_aspect": clip_target.flatten(0),
            "clip_tweet": clip_tweet.flatten(0),
            "candidate_input_ids": torch.stack(candidate_input_ids),
            "candidate_attention_mask": torch.stack(candidate_attention_mask),
            "labels": torch.tensor(label, dtype=torch.long),
            "output_image": output_image,
            "output_text": output_text
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 直接从缓存中取
        return self.samples[idx]



class TwitterDataset_base(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = cfg.tokenizer
        self.max_len = cfg.max_len

        # ====== 将所有数据一次性预处理好缓存到内存 ======
        self.samples = []
        print(f"🔄 正在预处理 {len(df)} 条样本...")

        for idx in range(len(df)):
            row = df.iloc[idx]
            sample = self._process_one(row)
            self.samples.append(sample)

        print(f"✅ 数据预处理完成，共缓存 {len(self.samples)} 条样本。")

    def _process_one(self, row):
        tweet = str(row["tweet_content"])
        ocr_text = str(row["ocr_text"])
        image_id = str(row["image_id"])
        target = str(row["target"])
        image_info = str(row["image_info"])
        label = int(row["sentiment"])
        candidate_aspects = eval(row["nouns"])

        # ---------- 文本拼接 ----------
        if ocr_text != "no text":
            tweet_all = tweet + " " + ocr_text
        else:
            tweet_all = tweet + ". " + image_info

        # ---------- 加载图像 ----------
        if "2017" in self.cfg.dataset:
            image_dir = os.path.join(
                "/home/zxp/learn/datasets/IJCAI2019_data/twitter2017_images/", image_id
            )
        else:
            image_dir = os.path.join(
                "/home/zxp/learn/datasets/IJCAI2019_data/twitter2015_images/", image_id
            )
        image = Image.open(image_dir).convert("RGB")
        image_tensor = self.cfg.clip_image_preprocess(image)

        # ---------- Clip 模型文本 ----------
        tweet_target = tweet.replace("$T$", target)
        clip_target = self.cfg.clip_text_tokenizer([target])
        clip_tweet = self.cfg.clip_text_tokenizer([tweet_target])

        # ---------- Bert 文本 ----------
        encoding = self.tokenizer.encode_plus(
            tweet,
            text_pair=target,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        encoding_all = self.tokenizer.encode_plus(
            tweet_all,
            text_pair=target,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation="longest_first",
        )

        # ---------- 候选方面词 ----------
        if len(candidate_aspects) > 3:
            candidate_aspects = random.sample(candidate_aspects, 3)

        candidate_input_ids = []
        candidate_attention_mask = []
        for asp in candidate_aspects:
            candidate_encoding = self.tokenizer.encode_plus(
                tweet_target.replace(asp, "$T$"),
                text_pair=asp,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation="longest_first",
            )
            candidate_input_ids.append(candidate_encoding["input_ids"].flatten())
            candidate_attention_mask.append(candidate_encoding["attention_mask"].flatten())

        # 如果不足 3 个，填充
        while len(candidate_input_ids) < 3:
            candidate_input_ids.append(encoding["input_ids"].flatten())
            candidate_attention_mask.append(encoding["attention_mask"].flatten())

        return {
            "review_text": tweet,
            "targets": target,
            "image_id": image_id,
            "ocr_text": ocr_text,
            "image_info": image_info,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "input_ids_all": encoding_all["input_ids"].flatten(),
            "attention_mask_all": encoding_all["attention_mask"].flatten(),
            "clip_image": image_tensor,
            "clip_aspect": clip_target.flatten(0),
            "clip_tweet": clip_tweet.flatten(0),
            "candidate_input_ids": torch.stack(candidate_input_ids),
            "candidate_attention_mask": torch.stack(candidate_attention_mask),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 直接从缓存中取
        return self.samples[idx]
