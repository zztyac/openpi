# 策略服务脚本 - 用于启动一个WebSocket服务器来提供机器人策略推理服务
# 这个脚本是理解整个项目的最佳入口点，因为它展示了如何加载和使用训练好的模型

import dataclasses
import enum
import logging
import socket

import tyro  

from openpi.policies import policy as _policy  # 策略基类和包装器
from openpi.policies import policy_config as _policy_config  # 策略配置和创建工具
from openpi.serving import websocket_policy_server  # WebSocket服务器实现
from openpi.training import config as _config  # 训练配置


class EnvMode(enum.Enum):
    """支持的机器人环境类型"""

    ALOHA = "aloha"  # ALOHA真实机器人
    ALOHA_SIM = "aloha_sim"  # ALOHA模拟器
    DROID = "droid"  # DROID机器人平台
    LIBERO = "libero"  # LIBERO基准测试环境


@dataclasses.dataclass
class Checkpoint:
    """从训练好的检查点加载策略"""

    # 训练配置名称（例如："pi0_aloha_sim"）
    config: str
    # 检查点目录路径（例如："checkpoints/pi0_aloha_sim/exp/10000"）
    # 可以是本地路径或云存储路径（如 gs://）
    dir: str


@dataclasses.dataclass
class Default:
    """使用给定环境的默认策略"""


@dataclasses.dataclass
class Args:
    """serve_policy 脚本的命令行参数"""

    # 要服务的机器人环境类型（仅在使用默认策略时需要）
    env: EnvMode = EnvMode.ALOHA_SIM

    # 默认提示词：当数据中没有"prompt"键或模型没有默认提示词时使用
    # 例如："pick up the fork"（拿起叉子）
    default_prompt: str | None = None

    # WebSocket服务器监听的端口号
    port: int = 8000
    
    # 是否记录策略的行为用于调试（会保存推理记录）
    record: bool = False

    # 指定如何加载策略：可以是Checkpoint（自定义检查点）或Default（默认策略）
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# 每个环境对应的默认检查点配置
# 这些是预训练好的模型，存储在Google Cloud Storage上
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",  # 使用π₀.₅模型配置
        dir="gs://openpi-assets/checkpoints/pi05_base",  # 基础模型检查点
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",  # 使用π₀模型配置
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",  # ALOHA模拟器专用检查点
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",  # 使用π₀.₅模型配置
        dir="gs://openpi-assets/checkpoints/pi05_droid",  # DROID平台微调检查点
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",  # 使用π₀.₅模型配置
        dir="gs://openpi-assets/checkpoints/pi05_libero",  # LIBERO基准测试检查点
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """为给定环境创建默认策略
    
    Args:
        env: 机器人环境类型
        default_prompt: 可选的默认提示词
        
    Returns:
        加载好的策略对象
    """
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        # 核心步骤：
        # 1. 获取训练配置：_config.get_config(checkpoint.config)
        # 2. 从检查点创建训练好的策略：create_trained_policy()
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """根据命令行参数创建策略
    
    Args:
        args: 命令行参数对象
        
    Returns:
        策略对象（Policy实例）
    """
    match args.policy:
        case Checkpoint():
            # 情况1：从指定的检查点加载策略
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            # 情况2：使用环境的默认策略
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    """主函数：启动策略服务器
    
    工作流程：
    1. 创建策略对象
    2. 可选：包装策略记录器
    3. 创建WebSocket服务器
    4. 启动服务器并持续运行
    """
    # 步骤1：根据参数创建策略
    policy = create_policy(args)
    policy_metadata = policy.metadata  # 获取策略元数据（输入/输出规格等）

    # 步骤2：如果启用记录模式，用PolicyRecorder包装策略
    # PolicyRecorder会记录所有的输入观测和输出动作，用于调试和分析
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # 步骤3：获取主机信息并打印
    hostname = socket.gethostname()  # 获取主机名
    local_ip = socket.gethostbyname(hostname)  # 获取本地IP地址
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # 步骤4：创建WebSocket服务器
    # - host="0.0.0.0" 表示监听所有网络接口（允许远程连接）
    # - port 是监听端口（默认8000）
    # - metadata 包含策略的输入输出规格，客户端可以查询
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    
    # 步骤5：启动服务器并永久运行（直到手动停止）
    # 服务器会等待客户端连接，接收观测数据，返回动作预测
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # 使用tyro解析命令行参数并运行主函数
    # 例如：python serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=./checkpoints/...
    main(tyro.cli(Args))
