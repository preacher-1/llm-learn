from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 加载数据集
        self.dataset = load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # 读取到的是jsonl的某一行
        # 最后要返回input_ids, attention_mask，labels
        data = self.dataset[idx]
        tokens = self.tokenizer(
            str(data["text"]),
            add_special_tokens=False,  # 后续手动添加
            max_length=self.max_length - 2,  # 留出特殊标记的位置
            truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (
            self.max_length - len(tokens)
        )
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # attention_mask: 1 表示真实 token，0 表示 padding
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # labels 用于因果 LM：与 input_ids 相同，但 pad 部分置为 -100（loss 忽略）
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        
