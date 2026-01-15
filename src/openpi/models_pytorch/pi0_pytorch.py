"""
PI0 PyTorch模型实现

这是PI0/PI05视觉-语言-动作(VLA)模型的PyTorch实现。
模型架构：
1. PaliGemma: 处理图像和语言输入（视觉编码器 + 语言模型）
2. Expert Gemma: 处理动作生成（专家语言模型）
3. Flow Matching: 使用流匹配进行动作去噪和生成

主要组件：
- 图像编码：SigLIP视觉编码器
- 语言编码：Gemma语言模型
- 动作生成：基于流匹配的去噪过程
"""

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """获取设备安全的数据类型
    
    CPU不支持bfloat16，需要回退到float32
    """
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """计算标量位置的正弦-余弦位置编码向量
    
    用于将时间步编码为高维向量，类似于Transformer的位置编码
    
    Args:
        time: 时间步张量，形状为(batch_size,)
        dimension: 编码维度（必须是偶数）
        min_period: 最小周期
        max_period: 最大周期
        device: 设备
        
    Returns:
        位置编码张量，形状为(batch_size, dimension)
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    # 创建频率向量
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # 计算外积
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    # 拼接sin和cos编码
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """从Beta分布采样
    
    用于采样训练时的时间步
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """创建2D注意力掩码
    
    从big_vision复制的实现。
    
    token可以关注到累积mask_ar小于或等于自己的有效输入token。
    这样mask_ar int[B, N]可以用来设置多种类型的注意力，例如：
    
      [[1 1 1 1 1 1]]: 纯因果注意力
      
      [[0 0 0 1 1 1]]: prefix-lm注意力。前3个token可以相互关注，
                       后3个token使用因果注意力
                       
      [[1 0 1 0 1 0 0 1 0 0]]: 4个块之间的因果注意力。
                                一个块的token可以关注所有之前的块和同一块的所有token
    
    Args:
        pad_masks: bool[B, N] 如果是输入的一部分则为true，如果是padding则为false
        att_masks: int32[B, N] 掩码，1表示之前的token不能依赖它，
                               0表示它与之前的token共享相同的注意力掩码
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # 累积和创建因果掩码
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    # 结合padding掩码
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """PI0/PI05 PyTorch模型主类
    
    这是一个视觉-语言-动作(VLA)模型，结合了：
    1. PaliGemma: 多模态编码器（图像 + 语言）
    2. Expert Gemma: 动作专家模型
    3. Flow Matching: 基于流匹配的动作生成
    
    模型工作流程：
    - 训练时：给定观测和动作，学习去噪过程
    - 推理时：从噪声开始，逐步去噪生成动作
    """
    
    def __init__(self, config):
        """初始化PI0模型
        
        Args:
            config: 模型配置，包含：
                - paligemma_variant: PaliGemma模型变体（如"gemma_2b"）
                - action_expert_variant: 动作专家模型变体（如"gemma_300m"）
                - pi05: 是否使用PI05版本（带AdaRMS）
                - dtype: 训练精度
                - action_dim: 动作维度
                - action_horizon: 动作序列长度
        """
        super().__init__()
        self.config = config
        self.pi05 = config.pi05  # 是否使用PI05版本

        # 获取PaliGemma和动作专家的配置
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 创建PaliGemma + Expert Gemma组合模型
        # PI05使用AdaRMS（自适应RMS归一化）用于动作专家
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],  # [PaliGemma, Expert]
            precision=config.dtype,
        )

        # 动作投影层：将32维动作映射到专家模型的隐藏维度
        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            # PI05版本：使用时间MLP用于AdaRMS条件
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # PI0版本：使用状态投影和动作-时间融合MLP
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # 设置高精度矩阵乘法并编译sample_actions以加速推理
        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # 初始化梯度检查点标志
        self.gradient_checkpointing_enabled = False

        # 验证transformers_replace是否正确安装
        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """启用梯度检查点以优化内存
        
        梯度检查点通过在反向传播时重新计算激活值来节省内存，
        代价是增加约30%的计算时间
        """
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """检查梯度检查点是否启用"""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """辅助方法：如果启用则应用梯度检查点
        
        在训练时使用梯度检查点，推理时直接调用函数
        """
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """辅助方法：为transformer准备4D注意力掩码
        
        将2D掩码扩展为4D并转换为注意力分数的加性掩码
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # 将True转换为0（可以关注），False转换为大负数（不能关注）
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """辅助方法：预处理观测数据
        
        将原始观测转换为模型输入格式
        """
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),  # 图像列表
            list(observation.image_masks.values()),  # 图像掩码列表
            observation.tokenized_prompt,  # 分词后的提示
            observation.tokenized_prompt_mask,  # 提示掩码
            observation.state,  # 机器人状态
        )

    def sample_noise(self, shape, device):
        """采样高斯噪声
        
        用于流匹配的初始噪声
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        """采样训练时间步
        
        使用Beta分布采样时间步，范围在[0.001, 1.0]
        """
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001  # 避免边界值
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """嵌入前缀：图像和语言输入
        
        使用SigLIP嵌入图像，使用嵌入层嵌入语言token，
        为PaliGemma transformer处理做准备。
        
        Args:
            images: 图像列表
            img_masks: 图像掩码列表
            lang_tokens: 语言token
            lang_masks: 语言掩码
            
        Returns:
            embs: 拼接的嵌入 [batch_size, seq_len, hidden_dim]
            pad_masks: padding掩码 [batch_size, seq_len]
            att_masks: 注意力掩码 [batch_size, seq_len]
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 处理图像
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            # 使用SigLIP编码图像
            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # 创建注意力掩码，使图像token可以相互关注
            att_masks += [0] * num_img_embs

        # 处理语言token
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            # 缩放嵌入（标准Transformer做法）
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像和语言输入之间的完全注意力
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # 拼接所有嵌入
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # 从拼接张量的第一维获取批次大小
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """嵌入后缀：状态、噪声动作和时间步
        
        为Expert Gemma处理准备嵌入。
        
        Args:
            state: 机器人状态
            noisy_actions: 噪声动作
            timestep: 时间步
            
        Returns:
            embs: 拼接的嵌入
            pad_masks: padding掩码
            att_masks: 注意力掩码
            adarms_cond: AdaRMS条件（仅PI05）
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # PI0版本：嵌入状态
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # 设置注意力掩码，使图像和语言输入不关注状态或动作
            att_masks += [1]

        # 使用正弦-余弦位置编码嵌入时间步，敏感范围为[0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # 使用MLP融合时间步 + 动作信息
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            # PI0版本：拼接动作和时间嵌入
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # 应用MLP层
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu激活函数
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # PI05版本：使用时间MLP生成AdaRMS条件
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu激活函数
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb  # 用于AdaRMS的条件

        # 添加到输入token
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # 设置注意力掩码，使图像、语言和状态输入不关注动作token
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """训练前向传播：计算流匹配损失
        
        流匹配训练过程：
        1. 采样时间步t和噪声ε
        2. 计算噪声动作：x_t = t*ε + (1-t)*actions
        3. 计算速度场：u_t = ε - actions
        4. 模型预测速度场：v_t = model(observation, x_t, t)
        5. 计算MSE损失：loss = ||u_t - v_t||²
        
        Args:
            observation: 观测数据（图像、语言、状态）
            actions: 真实动作 [batch_size, action_horizon, action_dim]
            noise: 可选的噪声（用于调试）
            time: 可选的时间步（用于调试）
            
        Returns:
            损失张量 [batch_size, action_horizon, action_dim]
        """
        # 预处理观测
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # 采样噪声和时间步
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 流匹配：线性插值从噪声到动作
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 噪声动作
        u_t = noise - actions  # 真实速度场

        # 嵌入前缀（图像 + 语言）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        # 嵌入后缀（状态 + 噪声动作 + 时间）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        # 确保数据类型一致
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 拼接前缀和后缀的掩码
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 创建2D注意力掩码和位置ID
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 准备4D注意力掩码
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 应用梯度检查点（如果启用）
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 提取动作输出并投影回动作空间
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # 应用梯度检查点到最终动作投影（如果启用）
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)  # 预测的速度场

        # 计算MSE损失
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """推理：从噪声生成动作
        
        使用欧拉方法进行ODE求解，从t=1（纯噪声）到t=0（纯动作）
        
        Args:
            device: 设备
            observation: 观测数据
            noise: 初始噪声（可选）
            num_steps: 去噪步数（默认10步）
            
        Returns:
            生成的动作 [batch_size, action_horizon, action_dim]
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 预处理观测
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # 嵌入前缀并计算KV缓存（只需计算一次）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 计算图像和语言的key-value缓存
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 欧拉方法求解ODE：从t=1到t=0
        dt = -1.0 / num_steps  # 时间步长
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise  # 初始化为纯噪声
        time = torch.tensor(1.0, dtype=torch.float32, device=device)  # 从t=1开始
        
        # 迭代去噪
        while time >= -dt / 2:  # 直到t≈0
            expanded_time = time.expand(bsize)
            # 预测速度场
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # 欧拉步骤：x_{t+dt} = x_t + dt * v_t
            x_t = x_t + dt * v_t
            time += dt
        
        return x_t  # 返回去噪后的动作

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """应用一步去噪
        
        在给定时间步对噪声x_t进行一步去噪
        
        Args:
            state: 机器人状态
            prefix_pad_masks: 前缀padding掩码
            past_key_values: 缓存的KV（来自图像和语言）
            x_t: 当前噪声动作
            timestep: 当前时间步
            
        Returns:
            预测的速度场 v_t
        """
        # 嵌入后缀（状态 + 噪声动作 + 时间）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 创建前缀的2D padding掩码
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # 创建后缀的2D注意力掩码
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 拼接完整的注意力掩码
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 计算位置ID（考虑前缀偏移）
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 准备4D注意力掩码
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 前向传播（使用缓存的KV）
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # 提取动作输出并投影
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)  # 返回预测的速度场
