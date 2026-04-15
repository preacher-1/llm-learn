import os
import sys


__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse  # 命令行参数解析
import time  # 时间统计
import warnings  # 警告控制
import torch
import torch.distributed as dist  # 分布式训练支持
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载器

from model.model import MyModelConfig, MyModel
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import (  # 训练工具函数
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
)

# 忽略警告信息，保持输出清洁
warnings.filterwarnings("ignore")


def train_epoch(epoch, dataloader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step

    for step, data in enumerate(dataloader, start=start_step + 1):
        input_ids = data["input_ids"].to(args.device)
        labels = data["labels"].to(args.device)
        attention_mask = data["attention_mask"].to(args.device)

        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 前向传播
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()
        if step % args.gradient_accumulation_steps == 0:
            # 还原梯度
            scaler.unscale_(optimizer)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        # logging、评估与保存
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.gradient_accumulation_steps
            current_lr = optimizer.param_groups[0]["lr"]
            eta_min = spend_time * (iters - step) / max(1, step - start_step) // 60
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}] Step [{step}/{iters}] Loss: {current_loss:.4f} LR: {current_lr:.2e} epoch_time: {eta_min:.1f}min"
            )
            if wandb is not None:
                wandb.log(
                    {
                        "train_loss": current_loss,
                        "lr": current_lr,
                        "epoch_time": eta_min,
                    },
                    step=epoch * iters + step,
                )

        if (step % args.eval_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if model_config.use_moe else ""
            ckpt = f"{args.save_dir}/{args.save_weight}_{model_config.hidden_dim}{moe_suffix}_epoch{epoch + 1}_step{step}.pt"
            raw_model = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            raw_model = getattr(
                raw_model, "model", raw_model
            )  # 如果是包装了模型的分布式数据并行，获取原始模型

            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckpt)

            lm_checkpoint(
                model_config=model_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
            )
            model.train()
            del state_dict

        del input_ids, labels, attention_mask, outputs, loss

        if last_step > start_step and last_step % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=512, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"

    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)

    model_config = MyModelConfig(
        hidden_dim=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
    )

    resume_ckpt = None
    if args.from_resume:
        resume_ckpt = lm_checkpoint(
            lm_config=model_config,
            weight=args.from_weight,
            save_dir="../checkpoints",
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if args.dtype == "bfloat16" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    autocast_ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    wandb = None
    if args.use_wandb and is_main_process():
        import wandb

        wandb_id = resume_ckpt.get("wandb_id") if resume_ckpt else None
        resume = "must" if wandb_id else None

        run_name = (
            f"pretrain_Epoch_{args.epochs}_BS_{args.batch_size}_LR_{args.learning_rate}"
        )
        wandb.init(
            project=args.wandb_project, name=run_name, id=wandb_id, resume=resume
        )

    model, tokenizer = init_model(
        model_config, args.from_weight, device=args.device
    )  # 默认使用 model/ 下的minimind_tokenizer

    train_dataset = PretrainDataset(args.data_path, tokenizer, args.max_seq_len)

    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    scaler = torch.amp.GradScaler(enabled=("float16" in str(dtype)))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 如果恢复训练
    start_epoch, start_step = 0, 0
    if resume_ckpt:
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scaler.load_state_dict(resume_ckpt["scaler"])
        start_epoch = resume_ckpt.get("epoch", 0)
        start_step = resume_ckpt.get("step", 0)

    if args.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        Logger("Using torch.compile for acceleration")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if epoch == start_epoch and start_step > 0:
            # 如果是续训且有未完成的epoch，从上次的step继续训练
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_dataset)), args.batch_size, start_step
            )
            dataloader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            Logger(
                f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
            )
            train_epoch(
                epoch, dataloader, len(dataloader) + start_step, start_step, wandb
            )
        else:
            # 默认从头开始训练
            dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            train_epoch(epoch, dataloader, len(dataloader), 0, wandb)
