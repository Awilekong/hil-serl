"""
奖励分类器训练脚本

该脚本用于训练一个二分类器，用于区分成功和失败的转换（transitions）。
分类器可以用于强化学习中的奖励函数学习。

主要功能：
1. 加载成功和失败的转换数据
2. 使用数据增强技术训练分类器
3. 保存训练好的分类器检查点
"""

import glob
import os
import pickle as pkl
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


# 定义命令行参数
FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "实验名称，对应配置文件夹")
flags.DEFINE_integer("num_epochs", 150, "训练轮数")
flags.DEFINE_integer("batch_size", 256, "批次大小")


def main(_):
    """主训练函数"""
    # 检查实验名称是否在配置映射中
    assert FLAGS.exp_name in CONFIG_MAPPING, '未找到实验文件夹'
    # 获取实验配置
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    # 创建环境（fake_env=True表示不需要真实机器人）
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    # 获取JAX设备（GPU/TPU）并设置数据分片策略
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # 创建正样本（成功转换）的回放缓冲区
    pos_buffer = ReplayBuffer(
        env.observation_space,  # 观察空间
        env.action_space,       # 动作空间
        capacity=20000,         # 缓冲区容量
        include_label=True,     # 包含标签信息
    )

    # 加载所有成功样本数据文件
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    for path in success_paths:
        success_data = pkl.load(open(path, "rb"))
        for trans in success_data:
            # 跳过包含图像的转换（仅使用非图像观察）
            if "images" in trans['observations'].keys():
                continue
            # 为成功样本设置标签为1
            trans["labels"] = 1
            # 随机采样动作（动作本身不用于分类）
            trans['actions'] = env.action_space.sample()
            pos_buffer.insert(trans)
            
    # 创建正样本迭代器，每次采样batch_size的一半
    pos_iterator = pos_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,  # 正负样本各占一半
        },
        device=sharding.replicate(),  # 在所有设备上复制数据
    )
    
    # 创建负样本（失败转换）的回放缓冲区
    neg_buffer = ReplayBuffer(
        env.observation_space,  # 观察空间
        env.action_space,       # 动作空间
        capacity=50000,         # 负样本容量更大（通常失败样本更多）
        include_label=True,     # 包含标签信息
    )
    # 加载所有失败样本数据文件
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))
    for path in failure_paths:
        failure_data = pkl.load(
            open(path, "rb")
        )
        for trans in failure_data:
            # 跳过包含图像的转换
            if "images" in trans['observations'].keys():
                continue
            # 为失败样本设置标签为0
            trans["labels"] = 0
            # 随机采样动作
            trans['actions'] = env.action_space.sample()
            neg_buffer.insert(trans)
            
    # 创建负样本迭代器
    neg_iterator = neg_buffer.get_iterator(
        sample_args={
            "batch_size": FLAGS.batch_size // 2,  # 负样本也占一半
        },
        device=sharding.replicate(),
    )

    # 打印数据集大小信息
    print(f"失败样本缓冲区大小: {len(neg_buffer)}")
    print(f"成功样本缓冲区大小: {len(pos_buffer)}")

    # 初始化随机数生成器
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    # 采样一个batch用于初始化分类器
    pos_sample = next(pos_iterator)
    neg_sample = next(neg_iterator)
    sample = concat_batches(pos_sample, neg_sample, axis=0)

    # 创建分类器模型
    rng, key = jax.random.split(rng)
    classifier = create_classifier(
        key,                      # 随机密钥
        sample["observations"],   # 样本观察用于推断输入形状
        config.classifier_keys,   # 用于分类的观察键
    )

    def data_augmentation_fn(rng, observations):
        """数据增强函数：对图像观察应用随机裁剪
        
        Args:
            rng: JAX随机数生成器
            observations: 观察字典
            
        Returns:
            增强后的观察
        """
        # 对每个像素键（图像观察）应用随机裁剪
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, 
                        padding=4,           # 裁剪前先padding 4个像素
                        num_batch_dims=2     # 批次维度数
                    )
                }
            )
        return observations

    @jax.jit  # JIT编译以提升性能
    def train_step(state, batch, key):
        """单步训练函数
        
        Args:
            state: 模型训练状态
            batch: 训练批次数据
            key: 随机密钥（用于dropout）
            
        Returns:
            更新后的状态、损失值、训练准确率
        """
        def loss_fn(params):
            """计算二元交叉熵损失"""
            # 前向传播获取logits
            logits = state.apply_fn(
                {"params": params}, batch["observations"], 
                rngs={"dropout": key}, train=True
            )
            # 计算sigmoid二元交叉熵损失
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        # 计算损失和梯度
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        # 计算训练准确率（不使用dropout）
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], 
            train=False, rngs={"dropout": key}
        )
        # 将sigmoid输出转换为二分类预测并计算准确率
        train_accuracy = jnp.mean((nn.sigmoid(logits) >= 0.5) == batch["labels"])

        # 应用梯度更新并返回
        return state.apply_gradients(grads=grads), loss, train_accuracy

    # 训练循环
    for epoch in tqdm(range(FLAGS.num_epochs)):
        # 从正负样本迭代器中各采样一半数据（保持类别平衡）
        pos_sample = next(pos_iterator)
        neg_sample = next(neg_iterator)
        
        # 合并正负样本批次
        batch = concat_batches(
            pos_sample, neg_sample, axis=0
        )
        
        # 对观察应用数据增强
        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        
        # 更新batch，添加增强后的观察和调整标签维度
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],  # 添加一个维度以匹配模型输出
            }
        )
        
        # 执行一步训练
        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        # 打印训练进度
        print(
            f"轮次: {epoch+1}, 训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.4f}"
        )

    # 保存训练好的分类器检查点
    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),  # 检查点保存目录
        classifier,                                       # 分类器状态
        step=FLAGS.num_epochs,                           # 训练步数
        overwrite=True,                                  # 覆盖已有检查点
    )
    print(f"\n分类器检查点已保存到: {os.path.join(os.getcwd(), 'classifier_ckpt/')}")
    

if __name__ == "__main__":
    app.run(main)