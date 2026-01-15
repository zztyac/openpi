"""
PyTorch训练脚本 - 支持PI0/PI05模型的单GPU、多GPU和多节点(DDP)训练

这个脚本是JAX训练器(`scripts/train.py`)的PyTorch版本，使用`PI0Pytorch`模型
以及现有的配置和数据管道（来自`src/openpi/training/config.py`和`src/openpi/training/data_loader.py`）

使用方法：

单GPU训练:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  示例:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # 从最新检查点恢复

多GPU训练(单节点):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  示例:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

多节点训练:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc  # 垃圾回收，用于内存管理
import logging
import os
import platform
import shutil
import time

import jax  # 用于数据处理的树结构操作
import numpy as np
import safetensors.torch  # 安全的模型权重保存/加载格式
import torch
import torch.distributed as dist  # 分布式训练支持
import torch.nn.parallel
import tqdm  # 进度条
import wandb  # 实验跟踪和可视化

import openpi.models.pi0_config  # 模型配置
import openpi.models_pytorch.pi0_pytorch  # PyTorch版本的PI0模型
import openpi.shared.normalize as _normalize  # 数据归一化工具
import openpi.training.config as _config  # 训练配置
import openpi.training.data_loader as _data  # 数据加载器


def init_logging():
    """初始化日志系统，设置自定义格式"""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    # 日志格式：时间 [级别] 消息 (进程ID:文件名:行号)
    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """初始化Weights & Biases实验跟踪
    
    Args:
        config: 训练配置
        resuming: 是否从之前的运行恢复
        enabled: 是否启用wandb（可以禁用用于调试）
    """
    if not enabled:
        wandb.init(mode="disabled")  # 禁用wandb，不上传任何数据
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        # 恢复训练：读取之前保存的wandb运行ID，继续同一个实验
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        # 新训练：创建新的wandb运行，并保存运行ID以便后续恢复
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    """设置分布式数据并行(DDP)训练环境
    
    Returns:
        use_ddp: 是否使用DDP
        local_rank: 当前进程的本地rank（GPU编号）
        device: 当前进程使用的设备
    """
    # 从环境变量获取总进程数（由torchrun设置）
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1  # 多于1个进程时启用DDP
    
    if use_ddp and not torch.distributed.is_initialized():
        # 初始化进程组：NCCL用于GPU，GLOO用于CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # 启用DDP调试信息，帮助排查分布式训练问题
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    # 获取本地rank（当前节点上的GPU编号）
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)  # 设置当前进程使用的GPU
    return use_ddp, local_rank, device


def cleanup_ddp():
    """清理DDP资源，在训练结束时调用"""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 等待所有进程完成
        torch.distributed.destroy_process_group()  # 销毁进程组


def set_seed(seed: int, local_rank: int):
    """设置随机种子以确保可复现性
    
    每个进程使用不同的种子（seed + local_rank），确保数据增强的多样性
    """
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    """构建数据加载器
    
    使用统一的数据加载器，支持PyTorch框架
    """
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """从模型获取状态字典，处理DDP包装器
    
    DDP会将模型包装在DistributedDataParallel中，需要通过.module访问原始模型
    """
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """从模型获取参数，处理DDP包装器"""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """保存检查点，包含模型状态、优化器状态和元数据
    
    Args:
        model: 模型对象
        optimizer: 优化器对象
        global_step: 当前训练步数
        config: 训练配置
        is_main: 是否为主进程（只有主进程保存检查点）
        data_config: 数据配置（包含归一化统计信息）
    """
    if not is_main:
        return 

    # 只在指定间隔或最后一步保存
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # 创建临时目录用于原子性保存（避免保存过程中断导致损坏）
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # 删除已存在的临时目录并创建新的
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型状态（使用safetensors格式，更安全）
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # 保存优化器状态（使用PyTorch原生格式）
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # 保存训练元数据（避免保存完整配置以防止JAX/Flax兼容性问题）
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # 保存归一化统计信息（用于推理时的数据预处理）
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # 原子性移动：将临时目录重命名为最终目录
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # 记录检查点到wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """加载最新的检查点并返回全局步数
    
    Args:
        model: 模型对象
        optimizer: 优化器对象
        checkpoint_dir: 检查点目录
        device: 设备（CPU或GPU）
        
    Returns:
        global_step: 恢复的训练步数
    """
    # 查找所有有效的检查点（数字命名的目录，排除临时目录）
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    # 使用最新的检查点
    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # 加载前清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # 加载模型状态
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # 加载优化器状态
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # 加载元数据
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # 内存不足错误处理
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """从检查点目录获取最新的检查点步数"""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """记录详细的GPU内存使用信息
    
    Args:
        device: GPU设备
        step: 当前步数
        phase: 当前阶段（用于标识日志）
    """
    if not torch.cuda.is_available():
        return

    # 获取内存使用统计
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # 转换为GB
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # 获取更详细的内存统计信息
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocd = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # 获取DDP信息（如果使用分布式训练）
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    """主训练循环
    
    这是训练的核心函数，包含以下步骤：
    1. 设置分布式训练环境
    2. 初始化wandb和检查点目录
    3. 构建数据加载器
    4. 创建模型和优化器
    5. 执行训练循环
    6. 保存检查点
    """
    # 步骤1：设置DDP环境
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)  # 判断是否为主进程
    set_seed(config.seed, local_rank)  # 设置随机种子

    # 步骤2：初始化检查点目录和wandb
    resuming = False
    if config.resume:
        # 查找基于实验名称的检查点目录
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # 验证并找到最新的有效检查点
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        # 如果设置了覆盖标志，删除现有检查点目录
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # 创建实验专用的检查点目录
    if not resuming:
        # 新训练：创建实验专用的检查点目录
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # 恢复训练：checkpoint_dir已经设置为实验目录
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # 初始化wandb（仅在主进程）
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 步骤3：构建数据加载器
    # 计算每个GPU的有效批次大小（用于DDP）
    # 对于N个GPU，每个GPU应该获得batch_size/N个样本，所以所有GPU的总批次大小是batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # 将原始批次大小传递给数据加载器 - 它会在内部处理DDP分割
    loader, data_config = build_datasets(config)

    # 在第一个批次时记录样本图像到wandb
    if is_main and config.wandb_enabled and not resuming:
        # 创建单独的数据加载器用于样本批次，避免消耗主加载器
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        
        # 将观测和动作转换为torch张量
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # 为wandb创建样本图像
        images_to_log = []
        # 从第一个图像张量获取批次大小
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # 将该批次项的所有相机视图水平拼接
            # 从NCHW格式转换为NHWC格式（wandb需要）
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # 积极清理样本批次的内存
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # 同时删除样本数据加载器
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # 步骤4：构建模型
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # 如果需要，将dataclass转换为Pi0Config
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # 更新dtype以匹配pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    # 启用梯度检查点以优化内存（以计算时间换内存）
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # 记录模型创建后的初始内存使用
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 为大规模训练启用内存优化（8+GPU）
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True  # 允许TF32精度（更快）
        torch.backends.cudnn.allow_tf32 = True
        # 设置内存分配配置
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    # 如果使用DDP，包装模型
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # 查找未使用的参数（调试用）
            gradient_as_bucket_view=True,  # 启用以提高内存效率
            static_graph=world_size >= 8,  # 8+GPU时启用静态图优化
        )

    # 如果指定了权重路径，加载预训练权重（用于微调）
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model), model_path
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # 步骤5：创建优化器和学习率调度器
    # 从配置获取学习率调度参数
    warmup_steps = config.lr_schedule.warmup_steps  # 预热步数
    peak_lr = config.lr_schedule.peak_lr  # 峰值学习率
    decay_steps = config.lr_schedule.decay_steps  # 衰减步数
    end_lr = config.lr_schedule.decay_lr  # 最终学习率

    # 使用配置参数创建AdamW优化器
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),  # Adam的beta参数
        eps=config.optimizer.eps,  # 数值稳定性的epsilon
        weight_decay=config.optimizer.weight_decay,  # L2正则化
    )

    # 如果恢复训练，加载检查点
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        """学习率调度函数：warmup + cosine衰减"""
        if step < warmup_steps:
            # 匹配JAX行为：从peak_lr / (warmup_steps + 1)开始
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # 余弦衰减
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    # 步骤6：开始训练
    model.train()  # 设置模型为训练模式
    start_time = time.time()
    infos = []  # 在日志间隔内收集统计信息
    
    # 打印训练配置信息
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # 创建进度条（仅主进程）
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    # 主训练循环 - 迭代直到达到num_train_steps
    while global_step < config.num_train_steps:
        # 为分布式训练设置epoch（确保每个epoch数据打乱不同）
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # 检查是否已达到目标步数
            if global_step >= config.num_train_steps:
                break

            # 统一数据加载器返回(observation, actions)元组
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            # 更新学习率
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 前向传播
            losses = model(observation, actions)
            # 确保losses是张量并处理不同的返回类型
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # 反向传播
            loss.backward()

            # 在反向传播后记录内存使用（前5步）
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 梯度裁剪（防止梯度爆炸）
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # 优化器步骤
            optim.step()
            optim.zero_grad(set_to_none=True)  # 清零梯度（set_to_none=True更高效）

            # 更积极地清理梯度（释放内存）
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # 收集统计信息
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            # 定期记录日志
            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # 计算日志间隔内的平均统计信息
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # 记录到wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # 重置统计信息收集

            global_step += 1
            # 使用新机制保存检查点
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # 关闭进度条
    if pbar is not None:
        pbar.close()

    # 结束wandb运行
    if is_main and config.wandb_enabled:
        wandb.finish()

    # 清理DDP资源
    cleanup_ddp()


def main():
    """主入口函数"""
    init_logging()  # 初始化日志系统
    config = _config.cli()  # 从命令行解析配置
    train_loop(config)  # 执行训练循环


if __name__ == "__main__":
    main()
