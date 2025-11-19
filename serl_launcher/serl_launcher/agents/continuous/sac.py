"""
软演员-评论家(Soft Actor-Critic, SAC)算法实现

该模块实现了SAC算法及其变体，用于连续控制任务的强化学习。
SAC通过最大化熵正则化的累积奖励来学习最优策略。

主要特点：
1. 最大熵强化学习：鼓励探索，提高鲁棒性
2. Off-policy学习：样本利用效率高
3. 自动温度调节：动态平衡探索与利用
4. Critic集成：减少Q值过估计

支持的算法变体：
- SAC (默认): 标准软演员-评论家算法
- TD3: Twin Delayed DDPG (设置固定标准差)
- REDQ: Randomized Ensembled Double Q-learning
- SAC-ensemble: 大型Critic集成

参考文献:
- Soft Actor-Critic (Haarnoja et al., 2018)
- REDQ (Chen et al., 2021)
"""

from functools import partial
from typing import Iterable, Optional, Tuple, FrozenSet

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from serl_launcher.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from serl_launcher.common.encoding import EncodingWrapper
from serl_launcher.common.optimizers import make_optimizer
from serl_launcher.common.typing import Batch, Data, Params, PRNGKey
from serl_launcher.networks.actor_critic_nets import Critic, Policy, ensemblize
from serl_launcher.networks.lagrange import GeqLagrangeMultiplier
from serl_launcher.networks.mlp import MLP
from serl_launcher.utils.train_utils import _unpack


class SACAgent(flax.struct.PyTreeNode):
    """SAC智能体：支持多种在线演员-评论家算法
    
    根据配置支持不同的算法变体：
     - SAC (默认): 标准软演员-评论家
     - TD3: policy_kwargs={"std_parameterization": "fixed", "fixed_std": 0.1}
     - REDQ: critic_ensemble_size=10, critic_subsample_size=2
     - SAC-ensemble: critic_ensemble_size>>1 (使用大型集成)
    
    Attributes:
        state: 训练状态，包含模型参数、优化器状态等
        config: 算法配置字典(不参与PyTree序列化)
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """Critic网络前向传播
        
        计算给定观察和动作的Q值估计。支持使用自定义参数进行梯度计算。
        
        Args:
            observations: 观察数据
            actions: 动作数据
            rng: 随机数生成器(用于dropout)
            grad_params: 可选的自定义参数(用于计算梯度)
            train: 是否处于训练模式
            
        Returns:
            Q值估计，形状为 [ensemble_size, batch_size]
        """
        if train:
            assert rng is not None, "训练时必须指定随机数生成器"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """目标Critic网络前向传播
        
        使用目标网络参数计算Q值，用于稳定训练。
        目标网络通过软更新(soft update)缓慢跟踪在线网络。
        
        Args:
            observations: 观察数据
            actions: 动作数据
            rng: 随机数生成器
            
        Returns:
            目标Q值估计
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    @jax.jit
    def jitted_forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """策略网络前向传播
        
        返回给定观察下的动作分布(通常是高斯分布)。
        在SAC中，策略是随机的，以最大化熵。
        
        Args:
            observations: 观察数据
            rng: 随机数生成器(用于dropout)
            grad_params: 可选的自定义参数
            train: 是否处于训练模式
            
        Returns:
            动作概率分布对象(distrax.Distribution)
        """
        if train:
            assert rng is not None, "训练时必须指定随机数生成器"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_temperature(
        self, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """获取当前温度参数
        
        温度参数α控制熵正则化的强度，平衡探索与利用。
        α越大，策略越随机(更多探索)；α越小，策略越确定(更多利用)。
        
        Args:
            grad_params: 可选的自定义参数
            
        Returns:
            当前温度值
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params}, name="temperature"
        )

    def temperature_lagrange_penalty(
        self, entropy: jnp.ndarray, *, grad_params: Optional[Params] = None
    ) -> distrax.Distribution:
        """计算温度的拉格朗日惩罚项
        
        使用拉格朗日乘数法自动调节温度，使策略熵保持在目标值附近。
        这实现了SAC的自动温度调节机制。
        
        Args:
            entropy: 当前策略的熵
            grad_params: 可选的自定义参数
            
        Returns:
            拉格朗日惩罚损失
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=entropy,
            rhs=self.config["target_entropy"],
            name="temperature",
        )

    def _compute_next_actions(self, batch, rng):
        """计算下一状态的动作和对数概率
        
        该函数被多个损失函数共享使用，避免重复计算。
        从策略网络采样下一状态的动作，用于计算目标Q值。
        
        Args:
            batch: 训练批次数据
            rng: 随机数生成器
            
        Returns:
            next_actions: 下一状态的采样动作
            next_actions_log_probs: 动作的对数概率
        """
        batch_size = batch["rewards"].shape[0]

        # 获取下一状态的动作分布
        next_action_distributions = self.forward_policy(
            batch["next_observations"], rng=rng
        )
        
        # 同时采样动作和计算对数概率(用于熵计算)
        (
            next_actions,
            next_actions_log_probs,
        ) = next_action_distributions.sample_and_log_prob(seed=rng)
        
        # 验证形状正确性
        chex.assert_equal_shape([batch["actions"], next_actions])
        chex.assert_shape(next_actions_log_probs, (batch_size,))

        return next_actions, next_actions_log_probs

    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """Critic网络损失函数
        
        使用Bellman方程计算TD误差，训练Critic准确预测Q值。
        支持Critic集成和子采样(REDQ)以减少过估计。
        
        计算流程：
        1. 采样下一动作和计算对数概率
        2. 使用目标Critic计算下一状态的Q值
        3. 可选：对Critic集成进行子采样(REDQ)
        4. 取最小Q值以减少过估计
        5. 计算目标Q值: r + γ * Q_target(s', a')
        6. 可选：添加熵正则化项
        7. 计算MSE损失
        
        Args:
            batch: 训练批次，包含 observations, actions, next_observations, rewards, masks
            params: Critic参数(用于计算梯度)
            rng: 随机数生成器
            
        Returns:
            critic_loss: Critic损失值
            info: 训练信息字典
        """
        batch_size = batch["rewards"].shape[0]
        
        # 采样下一动作
        rng, next_action_sample_key = jax.random.split(rng)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # 使用目标Critic计算下一状态的Q值(所有集成成员)
        # 目标网络只需前向传播，计算成本低
        target_next_qs = self.forward_target_critic(
            batch["next_observations"],
            next_actions,
            rng=rng,
        )  # 形状: (critic_ensemble_size, batch_size)

        # REDQ算法：对Critic集成进行子采样
        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            target_next_qs = target_next_qs[subsample_idcs]

        # 取集成中的最小Q值(减少过估计)
        target_next_min_q = target_next_qs.min(axis=0)
        chex.assert_shape(target_next_min_q, (batch_size,))

        # 计算目标Q值: r + γ * (1 - done) * Q_target(s', a')
        target_q = (
            batch["rewards"]
            + self.config["discount"] * batch["masks"] * target_next_min_q
        )
        chex.assert_shape(target_q, (batch_size,))

        # 可选：在目标中备份熵(backup entropy)
        # 这是一种变体，在目标中加入熵项以增强探索
        if self.config["backup_entropy"]:
            temperature = self.forward_temperature()
            target_q = target_q - temperature * next_actions_log_probs

        # 使用在线 Critic预测当前Q值
        predicted_qs = self.forward_critic(
            batch["observations"], batch["actions"], rng=rng, grad_params=params
        )

        chex.assert_shape(
            predicted_qs, (self.config["critic_ensemble_size"], batch_size)
        )
        
        # 将目标Q值广播到所有集成成员
        target_qs = target_q[None].repeat(self.config["critic_ensemble_size"], axis=0)
        chex.assert_equal_shape([predicted_qs, target_qs])
        
        # 计算均方误差(MSE)损失
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)

        # 收集训练信息
        info = {
            "critic_loss": critic_loss,
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_qs),
            "rewards": batch["rewards"].mean(),
        }

        return critic_loss, info

    def policy_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """策略网络损失函数
        
        最大化期望Q值减去熵惩罚项，即SAC的核心目标函数。
        这实现了最大熵强化学习(Maximum Entropy RL)。
        
        目标: max E[Q(s,a) - α*logπ(a|s)]
        - 第一项：最大化Q值(利用)
        - 第二项：最大化熵(探索)
        
        Args:
            batch: 训练批次
            params: 策略参数
            rng: 随机数生成器
            
        Returns:
            actor_loss: 策略损失(负目标函数)
            info: 训练信息
        """
        batch_size = batch["rewards"].shape[0]
        temperature = self.forward_temperature()

        # 分离随机数生成器用于不同目的
        rng, policy_rng, sample_rng, critic_rng = jax.random.split(rng, 4)
        
        # 从当前策略采样动作
        action_distributions = self.forward_policy(
            batch["observations"], rng=policy_rng, grad_params=params
        )
        actions, log_probs = action_distributions.sample_and_log_prob(seed=sample_rng)

        # 使用Critic评估采样动作的Q值
        predicted_qs = self.forward_critic(
            batch["observations"],
            actions,
            rng=critic_rng,
        )
        # 对集成取平均
        predicted_q = predicted_qs.mean(axis=0)
        chex.assert_shape(predicted_q, (batch_size,))
        chex.assert_shape(log_probs, (batch_size,))

        # SAC目标：Q值 - 温度 * 熵(即动作的负对数概率)
        actor_objective = predicted_q - temperature * log_probs
        # 损失是负目标(因为我们要最小化损失，即最大化目标)
        actor_loss = -jnp.mean(actor_objective)

        info = {
            "actor_loss": actor_loss,
            "temperature": temperature,
            "entropy": -log_probs.mean(),  # 熵是负对数概率的期望
        }

        return actor_loss, info

    def temperature_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """温度参数损失函数
        
        使用拉格朗日乘数法自动调节温度，使策略熵保持在目标值附近。
        
        约束：E[-logπ(a|s)] >= H_target
        即，策略熵应大于或等于目标熵。
        
        Args:
            batch: 训练批次
            params: 温度参数
            rng: 随机数生成器
            
        Returns:
            temperature_loss: 温度损失
            info: 训练信息
        """
        rng, next_action_sample_key = jax.random.split(rng)
        # 使用下一动作计算熵(与策略更新保持一致)
        next_actions, next_actions_log_probs = self._compute_next_actions(
            batch, next_action_sample_key
        )

        # 计算当前熵
        entropy = -next_actions_log_probs.mean()
        
        # 计算拉格朗日惩罚：如果熵低于目标，增大温度；反之减小
        temperature_loss = self.temperature_lagrange_penalty(
            entropy,
            grad_params=params,
        )
        return temperature_loss, {"temperature_loss": temperature_loss}
    
    def loss_fns(self, batch):
        """返回所有网络的损失函数
        
        将batch绑定到每个损失函数，以便在update中调用。
        
        Args:
            batch: 训练批次
            
        Returns:
            损失函数字典，键为网络名称
        """
        return {
            "critic": partial(self.critic_loss_fn, batch),
            "actor": partial(self.policy_loss_fn, batch),
            "temperature": partial(self.temperature_loss_fn, batch),
        }

    @partial(jax.jit, static_argnames=("pmap_axis", "networks_to_update"))
    def update(
        self,
        batch: Batch,
        *,
        pmap_axis: Optional[str] = None,
        networks_to_update: FrozenSet[str] = frozenset(
            {"actor", "critic", "temperature"}
        ),
        **kwargs
    ) -> Tuple["SACAgent", dict]:
        """对智能体的网络执行一步梯度更新
        
        该方法执行一个完整的SAC更新步骤，包括：
        1. 数据增强（如果配置）
        2. 计算各网络的损失和梯度
        3. 更新网络参数
        4. 软更新目标网络
        5. 更新随机数生成器
        
        在高UTD(Update-to-Data ratio)设置中，常见做法是：
        - 频繁更新Critic（如每步10次）
        - 较少更新Actor和Temperature（如每10步1次）
        
        Args:
            batch: 训练批次数据，必须包含键:
                "observations": 当前状态观察
                "actions": 执行的动作
                "next_observations": 下一状态观察
                "rewards": 获得的奖励
                "masks": 终止掩码(1-done)，episode结束时为0
            pmap_axis: 用于并行映射的轴（None表示不使用pmap）
            networks_to_update: 要更新的网络名称集合（默认：所有网络）
                例如：frozenset({"critic"}) 仅更新Critic
            **kwargs: 传递给损失函数的额外参数
            
        Returns:
            (new_agent, info): 
                - new_agent: 更新后的智能体
                - info: 包含损失值、学习率等训练信息的字典
        """
        batch_size = batch["rewards"].shape[0]
        chex.assert_tree_shape_prefix(batch, (batch_size,))

        # 解包批次数据（如果需要）
        if self.config["image_keys"][0] not in batch["next_observations"]:
            batch = _unpack(batch)
            
        # 应用数据增强（如图像随机裁剪）
        rng, aug_rng = jax.random.split(self.state.rng)
        if "augmentation_function" in self.config.keys() and self.config["augmentation_function"] is not None:
            batch = self.config["augmentation_function"](batch, aug_rng)

        # 添加奖励偏置（如果配置）
        batch = batch.copy(
            add_or_replace={"rewards": batch["rewards"] + self.config["reward_bias"]}
        )

        # 获取所有网络的损失函数
        loss_fns = self.loss_fns(batch, **kwargs)

        # 仅计算指定网络的梯度
        assert networks_to_update.issubset(
            loss_fns.keys()
        ), f"无效的网络名称: {networks_to_update}"
        # 对不需要更新的网络，将损失函数替换为返回0
        for key in loss_fns.keys() - networks_to_update:
            loss_fns[key] = lambda params, rng: (0.0, {})

        # 应用损失函数，计算梯度并更新参数
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # 软更新目标网络（如果Critic被更新）
        # 目标网络参数 = τ * 在线网络参数 + (1-τ) * 目标网络参数
        if "critic" in networks_to_update:
            new_state = new_state.target_update(self.config["soft_target_update_rate"])

        # 更新随机数生成器状态
        new_state = new_state.replace(rng=rng)

        # 记录各网络的学习率
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
        **kwargs,
    ) -> jnp.ndarray:
        """从策略网络采样动作
        
        使用外部随机数生成器采样动作，**不会更新智能体内部的RNG状态**。
        这使得该方法可以在推理时安全地并行调用。
        
        Args:
            observations: 输入观察
            seed: 外部随机数生成器（必须提供）
            argmax: 是否返回确定性动作（模式/均值）
                - True: 返回分布的mode（用于评估）
                - False: 从分布中随机采样（用于探索）
            **kwargs: 额外参数
            
        Returns:
            采样或确定性的动作数组
        """
        # 获取动作分布（不使用dropout）
        dist = self.forward_policy(observations, rng=seed, train=False)
        
        if argmax:
            # 返回确定性动作（分布的模式）
            return dist.mode()
        else:
            # 从分布中随机采样
            return dist.sample(seed=seed)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # 模型定义
        actor_def: nn.Module,
        critic_def: nn.Module,
        temperature_def: nn.Module,
        # 优化器配置
        actor_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        critic_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        temperature_optimizer_kwargs={
            "learning_rate": 3e-4,
        },
        # 算法配置
        discount: float = 0.95,
        soft_target_update_rate: float = 0.005,
        target_entropy: Optional[float] = None,
        entropy_per_dim: bool = False,
        backup_entropy: bool = False,
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        image_keys: Iterable[str] = None,
        augmentation_function: Optional[callable] = None,
        reward_bias: float = 0.0,
        **kwargs,
    ):
        """创建SAC智能体（通用版本）
        
        该方法用于创建完全自定义的SAC智能体，需要手动提供网络定义。
        对于基于图像的任务，推荐使用 `create_pixels` 方法。
        
        Args:
            rng: JAX随机数生成器
            observations: 示例观察（用于初始化网络形状）
            actions: 示例动作（用于初始化网络形状）
            
            # 网络定义
            actor_def: Actor网络模块
            critic_def: Critic网络模块
            temperature_def: 温度参数模块
            
            # 优化器
            actor_optimizer_kwargs: Actor优化器参数（learning_rate等）
            critic_optimizer_kwargs: Critic优化器参数
            temperature_optimizer_kwargs: Temperature优化器参数
            
            # 算法超参数
            discount: 折扣因子γ ∈ [0,1]，控制未来奖励的重要性
            soft_target_update_rate: 目标网络软更新率τ
                target = τ * online + (1-τ) * target
            target_entropy: 目标熵H_target（None则自动设为 -动作维度/2）
            entropy_per_dim: 是否按维度计算熵（未实现）
            backup_entropy: 是否在目标中备份熵
            critic_ensemble_size: Critic集成大小（减少过估计）
            critic_subsample_size: REDQ子采样大小（None表示使用全部）
            image_keys: 图像观察的键名列表
            augmentation_function: 数据增强函数
            reward_bias: 奖励偏置（添加到所有奖励）
            **kwargs: 其他配置参数
            
        Returns:
            初始化好的SACAgent实例
        """
        # 组合所有网络
        networks = {
            "actor": actor_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        model_def = ModuleDict(networks)

        # 定义优化器
        txs = {
            "actor": make_optimizer(**actor_optimizer_kwargs),
            "critic": make_optimizer(**critic_optimizer_kwargs),
            "temperature": make_optimizer(**temperature_optimizer_kwargs),
        }

        # 初始化网络参数
        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[observations],
            critic=[observations, actions],
            temperature=[],
        )["params"]

        # 创建训练状态
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,  # 初始时目标网络=在线网络
            rng=create_rng,
        )

        # 配置算法参数
        assert not entropy_per_dim, "entropy_per_dim未实现"
        if target_entropy is None:
            # 默认目标熵：负的动作维度的一半
            # 这是SAC论文中的启发式设置
            target_entropy = -actions.shape[-1] / 2

        return cls(
            state=state,
            config=dict(
                critic_ensemble_size=critic_ensemble_size,
                critic_subsample_size=critic_subsample_size,
                discount=discount,
                soft_target_update_rate=soft_target_update_rate,
                target_entropy=target_entropy,
                backup_entropy=backup_entropy,
                image_keys=image_keys,
                reward_bias=reward_bias,
                augmentation_function=augmentation_function,
                **kwargs,
            ),
        )

    @classmethod
    def create_pixels(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # 模型架构
        encoder_type: str = "resnet-pretrained",
        use_proprio: bool = False,
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "uniform",
        },
        critic_ensemble_size: int = 2,
        critic_subsample_size: Optional[int] = None,
        temperature_init: float = 1.0,
        image_keys: Iterable[str] = ("image",),
        augmentation_function: Optional[callable] = None,
        **kwargs,
    ):
        """创建基于像素输入的SAC智能体（推荐方法）
        
        这是创建视觉强化学习智能体的便捷方法，自动配置：
        - 视觉编码器（ResNet-10）
        - Actor和Critic网络
        - 温度参数
        
        适用于机器人操作等需要处理图像输入的任务。
        
        Args:
            rng: JAX随机数生成器
            observations: 示例观察（包含图像）
            actions: 示例动作
            
            # 编码器配置
            encoder_type: 编码器类型
                - "resnet": 可训练的ResNet-10
                - "resnet-pretrained": 预训练ResNet-10（推荐）
            use_proprio: 是否使用本体感受信息（关节位置、速度等）
            
            # 网络架构
            critic_network_kwargs: Critic MLP配置（隐藏层维度等）
            policy_network_kwargs: Policy MLP配置
            policy_kwargs: 策略分布配置
                - tanh_squash_distribution: 是否使用tanh压缩到[-1,1]
                - std_parameterization: 标准差参数化方式
            
            # 算法配置
            critic_ensemble_size: Critic集成大小（通常2-10）
            critic_subsample_size: REDQ子采样大小（None或2）
            temperature_init: 温度初始值
            image_keys: 图像观察键名（如 ("wrist_image", "overhead_image")）
            augmentation_function: 图像增强函数（如随机裁剪）
            **kwargs: 其他参数传递给 `create` 方法
            
        Returns:
            配置好的SACAgent实例
        """
        # 启用网络最后一层激活函数
        policy_network_kwargs["activate_final"] = True
        critic_network_kwargs["activate_final"] = True

        # 根据编码器类型创建视觉编码器
        if encoder_type == "resnet":
            # 可训练的ResNet-10
            from serl_launcher.vision.resnet_v1 import resnetv1_configs

            encoders = {
                image_key: resnetv1_configs["resnetv1-10"](
                    pooling_method="spatial_learned_embeddings",  # 空间学习嵌入
                    num_spatial_blocks=8,
                    bottleneck_dim=256,  # 瓶颈层维度
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        elif encoder_type == "resnet-pretrained":
            # 预训练ResNet-10（冻结特征提取，仅训练头部）
            from serl_launcher.vision.resnet_v1 import (
                PreTrainedResNetEncoder,
                resnetv1_configs,
            )

            # 冻结的预训练编码器
            pretrained_encoder = resnetv1_configs["resnetv1-10-frozen"](
                pre_pooling=True,  # 在池化前返回特征
                name="pretrained_encoder",
            )
            encoders = {
                image_key: PreTrainedResNetEncoder(
                    pooling_method="spatial_learned_embeddings",
                    num_spatial_blocks=8,
                    bottleneck_dim=256,
                    pretrained_encoder=pretrained_encoder,
                    name=f"encoder_{image_key}",
                )
                for image_key in image_keys
            }
        else:
            raise NotImplementedError(f"未知的编码器类型: {encoder_type}")

        # 创建编码器封装（处理多模态输入）
        encoder_def = EncodingWrapper(
            encoder=encoders,
            use_proprio=use_proprio,  # 是否拼接本体感受信息
            enable_stacking=True,      # 启用帧堆叠
            image_keys=image_keys,
        )

        # 为Actor和Critic分配独立的编码器
        encoders = {
            "critic": encoder_def,
            "actor": encoder_def,
        }

        # 定义Critic网络
        critic_backbone = partial(MLP, **critic_network_kwargs)
        # 创建Critic集成（多个Critic并行训练）
        critic_backbone = ensemblize(critic_backbone, critic_ensemble_size)(
            name="critic_ensemble"
        )
        critic_def = partial(
            Critic, encoder=encoders["critic"], network=critic_backbone
        )(name="critic")

        # 定义Policy网络
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],  # 动作维度
            **policy_kwargs,
            name="actor",
        )

        # 定义温度参数（使用拉格朗日乘数法）
        temperature_def = GeqLagrangeMultiplier(
            init_value=temperature_init,  # 初始温度
            constraint_shape=(),          # 标量约束
            constraint_type="geq",        # 大于等于约束
            name="temperature",
        )

        # 调用通用create方法
        agent = cls.create(
            rng,
            observations,
            actions,
            actor_def=policy_def,
            critic_def=critic_def,
            temperature_def=temperature_def,
            critic_ensemble_size=critic_ensemble_size,
            critic_subsample_size=critic_subsample_size,
            image_keys=image_keys,
            augmentation_function=augmentation_function,
            **kwargs,
        )

        # 如果使用预训练编码器，加载预训练权重
        if "pretrained" in encoder_type:
            from serl_launcher.utils.train_utils import load_resnet10_params
            agent = load_resnet10_params(agent, image_keys)

        return agent
