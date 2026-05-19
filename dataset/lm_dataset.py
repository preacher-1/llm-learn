from torch.utils.data import Dataset
import os
from datasets import load_dataset

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, num_proc=8):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 1. 加载原始数据集
        # 注意：第一次处理后，HuggingFace 会自动在 ~/.cache/huggingface/datasets 建立缓存
        # 之后哪怕重新启动训练，也会秒读缓存，不会重新走分词逻辑
        raw_dataset = load_dataset("json", data_files=data_path, split="train")

        # 2. 定义核心的“离线分词与拼接 (Packing)”函数
        def tokenize_and_pack(examples):
            # 批量进行 Tokenize，不截断、不加特殊字符（我们自己控制）
            tokens_dict = self.tokenizer(
                examples["text"],
                add_special_tokens=False,
            )

            # 将这个批次里所有的文章首尾相连，拼成一个超长的一维数组
            concatenated_ids = []
            for ids in tokens_dict["input_ids"]:
                # 每一篇文章的格式：[BOS] + 内容 + [EOS]
                seq = (
                    [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
                )
                concatenated_ids.extend(seq)

            # 计算总长度，并去掉尾部不够 max_length 长度的零头数据
            total_length = len(concatenated_ids)
            total_length = (total_length // self.max_length) * self.max_length

            # 像切香肠一样，严格按 max_length 切分为固定长度的块
            result_input_ids = [
                concatenated_ids[i : i + self.max_length]
                for i in range(0, total_length, self.max_length)
            ]

            # Labels 和 Input_ids 完全对齐，没有 Pad，全都是有效预测
            result_labels = [row.copy() for row in result_input_ids]

            # 因为序列全满，所以 Attention Mask 全是 1
            result_attention_mask = [
                [1] * self.max_length for _ in range(len(result_input_ids))
            ]

            return {
                "input_ids": result_input_ids,
                "attention_mask": result_attention_mask,
                "labels": result_labels,
            }

        # 3. 使用 map 映射整个数据集
        # batched=True 会批量将数据送入 tokenize_and_pack，并且允许输出与输入的行数不一致
        self.dataset = raw_dataset.map(
            tokenize_and_pack,
            batched=True,
            num_proc=num_proc,  # 开启多进程加速处理
            remove_columns=raw_dataset.column_names,  # 移除原始的 'text' 列
            desc="Running tokenizer and packing sequences",
        )

        # 4. 直接在底层转换为 PyTorch 张量格式，提升 DataLoader 抓取速度
        self.dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        # 这里发生了质变：之前这里有大量的 CPU 切片和逻辑判断
        # 现在仅仅是从 Arrow 缓存中取出一个切片好的 Tensor 字典，速度极快
        return self.dataset[idx]
