""" 
Actor-Critic网络架构

该模块实现了强化学习中的Actor-Critic网络架构，包括：
- ValueCritic: 状态价值函数V(s)
- Critic: 动作价值函数Q(s,a)
- GraspCritic: 离散抓取动作的Q函数
- Policy: 策略网络π(a|s)
- TanhMultivariateNormalDiag: Tanh压缩的多元正态分布

这些网络是SAC等算法的核心组件。
"""

from typing import Optional
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import default_init


class ValueCritic(nn.Module):
    """状态价值函数V(s)网络
    
    估计状态的价值，不依赖于具体动作。
    用于某些算法（如A2C、PPO）中的优势函数计算。
    
    Attributes:
        encoder: 观察编码器（如视觉编码器）
        network: 主干网络（通常是MLP）
        init_final: 最终层初始化范围（None则使用默认初始化）
    """
    encoder: nn.Module
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """计算状态价值V(s)
        
        Args:
            observations: 状态观察
            train: 是否处于训练模式
            
        Returns:
            状态价值标量，形状为 [batch_size]
        """
        # 编码观察 -> 主干网络 -> 输出标量
        outputs = self.network(self.encoder(observations), train)
        
        # 最终层：输出单个价值
        if self.init_final is not None:
            # 使用小范围均匀初始化（有助于训练稳定性）
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        
        # 移除最后一个维度 [batch_size, 1] -> [batch_size]
        return jnp.squeeze(value, -1)


def multiple_action_q_function(forward):
    """多动作Q函数装饰器
    
    允许Q函数同时评估多个动作，用于某些算法（如CQL）。
    
    支持两种输入模式：
    - 2D动作 [batch_size, action_dim]: 标准模式，每个状态一个动作
    - 3D动作 [batch_size, num_actions, action_dim]: 每个状态多个动作
    
    对于3D输入，使用vmap并行计算所有动作的Q值。
    
    Args:
        forward: 原始的Q函数前向传播方法
        
    Returns:
        包装后的方法，支持批量动作评估
    """
    def wrapped(self, observations, actions, train=False):
        if jnp.ndim(actions) == 3:
            # 3D输入：[batch, num_actions, action_dim]
            # 使用vmap沿num_actions维度并行计算
            q_values = jax.vmap(
                lambda a: forward(self, observations, a, train),
                in_axes=1,      # 沿着第1维（num_actions）映射
                out_axes=-1,    # 输出沿最后一维堆叠
            )(actions)
        else:
            # 2D输入：标准的 [batch, action_dim]
            q_values = forward(self, observations, actions, train)
        return q_values

    return wrapped


class Critic(nn.Module):
    """动作价值函数Q(s,a)网络
    
    估计在状态s下执行动作a的价值。
    这是SAC、TD3等off-policy算法的核心组件。
    
    网络结构：
    1. 编码器编码观察（可选）
    2. 拼接编码后的观察和动作
    3. MLP处理拼接后的特征
    4. 输出单个Q值
    
    通常与ensemblize结合使用创建Q值集成，以减少过估计。
    
    Attributes:
        encoder: 观察编码器（None则直接使用原始观察）
        network: 主干网络（通常是MLP）
        init_final: 最终层初始化范围
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None

    @nn.compact
    @multiple_action_q_function  # 支持多动作批量评估
    def __call__(
        self, observations: jnp.ndarray, actions: jnp.ndarray, train: bool = False
    ) -> jnp.ndarray:
        """计算动作价值Q(s,a)
        
        Args:
            observations: 状态观察
            actions: 动作，形状为 [batch_size, action_dim] 或 [batch_size, num_actions, action_dim]
            train: 是否处于训练模式
            
        Returns:
            Q值，形状为 [batch_size] 或 [batch_size, num_actions]
        """
        # 编码观察（如果有编码器）
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)

        # 拼接观察编码和动作作为Q网络输入
        inputs = jnp.concatenate([obs_enc, actions], -1)
        
        # 通过主干网络处理
        outputs = self.network(inputs, train)
        
        # 最终层：输出Q值
        if self.init_final is not None:
            value = nn.Dense(
                1,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(1, kernel_init=default_init())(outputs)
        
        # 移除最后一个维度
        return jnp.squeeze(value, -1)

    
class GraspCritic(nn.Module):
    """抓取动作的Q函数网络（用于混合动作空间）
    
    专门用于评估离散抓取动作的Q值，采用DQN风格。
    与标准Critic不同，该网络不以动作作为输入，而是为每个离散动作输出一个Q值。
    
    应用场景：
    - 单臂机器人：output_dim=3，对应抓取器的3种状态 {-1, 0, 1}
    - 双臂机器人：output_dim=9，对应两个抓取器的组合 3×3=9
    
    Attributes:
        encoder: 观察编码器（可选）
        network: 主干网络
        init_final: 最终层初始化范围
        output_dim: 输出维度（离散动作数量）
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    init_final: Optional[float] = None
    output_dim: Optional[int] = 3  # 默认3个离散抓取动作
    
    @nn.compact
    def __call__(
        self, 
        observations: jnp.ndarray, 
        train: bool = False
    ) -> jnp.ndarray:
        """计算所有离散抓取动作的Q值
        
        Args:
            observations: 状态观察
            train: 是否处于训练模式
            
        Returns:
            Q值向量，形状为 [batch_size, output_dim]
            每个元素对应一个离散抓取动作的Q值
        """
        # 编码观察
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations)
        
        # 通过主干网络
        outputs = self.network(obs_enc, train)
        
        # 输出层：为每个离散动作输出一个Q值
        if self.init_final is not None:
            value = nn.Dense(
                self.output_dim,
                kernel_init=nn.initializers.uniform(-self.init_final, self.init_final),
            )(outputs)
        else:
            value = nn.Dense(self.output_dim, kernel_init=default_init())(outputs)
        
        return value  # [batch_size, output_dim]


def ensemblize(cls, num_qs, out_axes=0):
    """创建网络集成的高阶函数
    
    将单个网络类转换为集成版本，创建多个独立参数化的网络副本。
    使用JAX的vmap实现高效的并行计算。
    
    集成的好处：
    1. 减少Q值过估计（TD3、SAC的关键技巧）
    2. 提供不确定性估计
    3. 提高样本效率（REDQ）
    
    典型用法：
        critic_ensemble = ensemblize(Critic, num_qs=10)
        
    Args:
        cls: 要集成的网络类
        num_qs: 集成中的网络数量
        out_axes: vmap的输出轴（0表示在第0维堆叠输出）
        
    Returns:
        集成模块类，输出形状为 [num_qs, batch_size, ...]
    """
    class EnsembleModule(nn.Module):
        @nn.compact
        def __call__(self, *args, train=False, **kwargs):
            # 使用vmap创建集成：并行运行num_qs个独立网络
            ensemble = nn.vmap(
                cls,
                variable_axes={"params": 0},  # 每个网络有独立参数
                split_rngs={"params": True, "dropout": True},  # 独立的随机数流
                in_axes=None,      # 输入在所有网络间共享
                out_axes=out_axes, # 输出沿指定轴堆叠
                axis_size=num_qs,  # 集成大小
            )
            # 前向传播
            return ensemble()(*args, **kwargs)

    return EnsembleModule

class Policy(nn.Module):
    """策略网络π(a|s)
    
    输出给定状态下的动作概率分布（通常是高斯分布）。
    这是SAC等算法中的Actor组件。
    
    支持多种配置：
    - 标准差参数化：exp、softplus、uniform、fixed
    - Tanh压缩：将动作压缩到[-1,1]范围
    - 温度缩放：控制探索程度
    
    Attributes:
        encoder: 观察编码器（可选）
        network: 主干网络
        action_dim: 动作维度
        init_final: 最终层初始化范围
        std_parameterization: 标准差参数化方式
            - "exp": log_std -> std = exp(log_std)
            - "softplus": std = softplus(x)
            - "uniform": 所有动作维度共享固定的可学习log_std
            - "fixed": 固定标准差（用于确定性策略如TD3）
        std_min: 标准差下限（避免数值不稳定）
        std_max: 标准差上限
        tanh_squash_distribution: 是否使用tanh压缩动作到[-1,1]
        fixed_std: 固定标准差值（仅当std_parameterization="fixed"时使用）
    """
    encoder: Optional[nn.Module]
    network: nn.Module
    action_dim: int
    init_final: Optional[float] = None
    std_parameterization: str = "exp"  # "exp", "softplus", "fixed", or "uniform"
    std_min: Optional[float] = 1e-5
    std_max: Optional[float] = 10.0
    tanh_squash_distribution: bool = False
    fixed_std: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0, train: bool = False, non_squash_distribution: bool = False,
    ) -> distrax.Distribution:
        """生成动作分布
        
        Args:
            observations: 状态观察
            temperature: 温度参数，控制探索程度
                - temperature > 1: 增加探索（更大的标准差）
                - temperature < 1: 减少探索（更小的标准差）
                - temperature = 1: 标准探索
            train: 是否处于训练模式
            non_squash_distribution: 是否强制不使用tanh压缩
            
        Returns:
            动作概率分布（高斯分布或Tanh压缩的高斯分布）
        """
        # 编码观察（停止梯度以避免影响编码器训练）
        if self.encoder is None:
            obs_enc = observations
        else:
            obs_enc = self.encoder(observations, train=train, stop_gradient=True)

        # 通过主干网络
        outputs = self.network(obs_enc, train=train)

        # 输出层：动作均值
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
        
        # 计算标准差（根据参数化方式）
        if self.fixed_std is None:
            if self.std_parameterization == "exp":
                # 指数参数化：网络输出log_std，取exp得到std
                # 优点：std始终为正，梯度流动好
                log_stds = nn.Dense(self.action_dim, kernel_init=default_init())(
                    outputs
                )
                stds = jnp.exp(log_stds)
            elif self.std_parameterization == "softplus":
                # Softplus参数化：std = log(1 + exp(x))
                # 优点：更平滑，避免exp的梯度爆炸
                stds = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)
                stds = nn.softplus(stds)
            elif self.std_parameterization == "uniform":
                # 均匀参数化：所有维度共享相同的可学习log_std
                # 优点：参数少，适合低维动作空间
                log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )
                stds = jnp.exp(log_stds)
            else:
                raise ValueError(
                    f"无效的std_parameterization: {self.std_parameterization}"
                )
        else:
            # 固定标准差（用于TD3等确定性策略）
            assert self.std_parameterization == "fixed"
            stds = jnp.array(self.fixed_std)

        # 裁剪标准差到合理范围，避免数值不稳定
        # 在最大熵RL中，最优std与√temperature成正比
        stds = jnp.clip(stds, self.std_min, self.std_max) * jnp.sqrt(temperature)

        # 创建动作分布
        if self.tanh_squash_distribution and not non_squash_distribution:
            # Tanh压缩分布：将动作压缩到[-1,1]范围
            # 适用于大多数机器人控制任务
            distribution = TanhMultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )
        else:
            # 标准多元高斯分布（无界）
            distribution = distrax.MultivariateNormalDiag(
                loc=means,
                scale_diag=stds,
            )

        return distribution
    
    def get_features(self, observations):
        """提取观察的编码特征
        
        用于可视化、分析或其他需要中间特征的任务。
        
        Args:
            observations: 状态观察
            
        Returns:
            编码后的特征向量
        """
        return self.encoder(observations, train=False, stop_gradient=True)


class TanhMultivariateNormalDiag(distrax.Transformed):
    """Tanh压缩的多元对角高斯分布
    
    通过Tanh变换将无界高斯分布映射到有界范围，适用于机器人控制等需要
    有界动作的任务。
    
    变换流程：
    1. 从高斯分布N(μ, σ)采样 x
    2. 应用tanh变换: y = tanh(x) ∈ (-1, 1)
    3. 可选：线性缩放到 [low, high] 范围
    
    重要：变换后需要修正概率密度（通过雅可比行列式）。
    
    参考文献：
    - Soft Actor-Critic (Haarnoja et al., 2018)
    """
    def __init__(
        self,
        loc: jnp.ndarray,
        scale_diag: jnp.ndarray,
        low: Optional[jnp.ndarray] = None,
        high: Optional[jnp.ndarray] = None,
    ):
        """初始化Tanh压缩高斯分布
        
        Args:
            loc: 高斯分布的均值μ
            scale_diag: 高斯分布的对角协方差矩阵的标准差σ
            low: 可选的动作下界（None则为-1）
            high: 可选的动作上界（None则为1）
        """
        # 基础分布：多元对角高斯
        distribution = distrax.MultivariateNormalDiag(loc=loc, scale_diag=scale_diag)

        layers = []

        # 如果指定了范围，添加缩放层
        if not (low is None or high is None):

            def rescale_from_tanh(x):
                """将tanh输出(-1,1)缩放到(low,high)"""
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                """计算缩放变换的对数雅可比行列式"""
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            # 添加线性缩放层
            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1,
                )
            )

        # 添加Tanh变换层（核心）
        layers.append(distrax.Block(distrax.Tanh(), 1))

        # 组合变换（注意：Chain是反向应用，所以Tanh先应用）
        bijector = distrax.Chain(layers)

        # 调用父类初始化：变换分布 = 基础分布 + 双射变换
        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        """返回分布的众数（最高概率密度点）
        
        对于高斯分布，mode=mean，因此直接变换均值。
        
        Returns:
            变换后的众数
        """
        return self.bijector.forward(self.distribution.mode())

    def stddev(self) -> jnp.ndarray:
        """返回变换后的标准差（近似）
        
        注意：这是近似值，因为Tanh变换后分布不再是高斯。
        
        Returns:
            变换后的标准差
        """
        return self.bijector.forward(self.distribution.stddev())
