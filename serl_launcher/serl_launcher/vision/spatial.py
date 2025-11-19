"""
空间学习嵌入模块

该模块实现了空间学习嵌入(Spatial Learned Embeddings)，用于将卷积特征图的空间信息
压缩为固定维度的特征向量。与全局平均池化不同，该方法通过学习空间权重来聚合特征。

应用场景：
- 机器人视觉中的位置感知特征提取
- 需要保留空间结构信息的特征压缩
"""
from typing import Sequence, Callable
import flax.linen as nn
import jax.numpy as jnp


class SpatialLearnedEmbeddings(nn.Module):
    """空间学习嵌入层
    
    通过学习的卷积核将空间特征图聚合为固定长度的向量。
    相比全局池化，该方法可以学习关注特定的空间位置。
    
    Attributes:
        height: 输入特征图的高度
        width: 输入特征图的宽度
        channel: 输入特征图的通道数
        num_features: 输出特征的数量（每个位置学习多少个特征）
        param_dtype: 参数数据类型
    """
    height: int
    width: int
    channel: int
    num_features: int = 5
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, features):
        """将空间特征图转换为学习的嵌入向量
        
        通过与学习的卷积核进行逐元素乘法和求和来聚合空间信息。
        
        Args:
            features: 输入特征图，形状为 [H, W, C] 或 [B, H, W, C]
                     B: batch size, H: height, W: width, C: channels
        
        Returns:
            嵌入向量，形状为 [num_features * channel] 或 [B, num_features * channel]
        """
        # 检测是否缺少batch维度
        squeeze = False
        if len(features.shape) == 3:
            # 添加batch维度以统一处理
            features = jnp.expand_dims(features, 0)
            squeeze = True

        # 初始化学习的空间权重核
        # 形状: [height, width, channel, num_features]
        # 每个空间位置和通道对应num_features个学习的权重
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),  # LeCun正态初始化
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        batch_size = features.shape[0]
        
        # 执行空间加权聚合
        # 1. expand_dims(features, -1): [B, H, W, C] -> [B, H, W, C, 1]
        # 2. expand_dims(kernel, 0): [H, W, C, F] -> [1, H, W, C, F]
        # 3. 逐元素相乘后沿空间维度求和: [B, H, W, C, F] -> [B, C, F]
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2)
        )
        
        # 展平为一维特征向量: [B, C, F] -> [B, C*F]
        features = jnp.reshape(features, [batch_size, -1])
        
        # 如果输入没有batch维度，则移除添加的batch维度
        if squeeze:
            features = jnp.squeeze(features, 0)
        
        return features
