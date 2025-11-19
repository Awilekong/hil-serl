"""
ResNet V1 编码器实现

该模块实现了ResNet V1架构及其变体，用于机器人视觉任务中的特征提取。
包含多种池化方法、空间注意力机制和条件化选项。

主要组件：
- ResNetBlock: 标准ResNet残差块
- BottleneckResNetBlock: 瓶颈结构残差块
- ResNetEncoder: 主编码器，支持多种配置
- SpatialSoftmax: 空间软最大值池化
- AddSpatialCoordinates: 添加空间坐标信息

参考文献:
- Deep Residual Learning for Image Recognition (He et al., 2016)
"""
import functools as ft
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from serl_launcher.vision.film_conditioning_layer import FilmConditioning
from serl_launcher.vision.data_augmentations import resize

ModuleDef = Any  # 模块定义类型别名


class AddSpatialCoordinates(nn.Module):
    """添加空间坐标信息到特征图
    
    在特征图的通道维度上附加归一化的(x, y)坐标信息。
    坐标范围为[-1, 1]，有助于网络学习位置相关的特征。
    
    Attributes:
        dtype: 坐标数据类型
    """
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """为输入特征图添加空间坐标通道
        
        Args:
            x: 输入特征图 [H, W, C] 或 [B, H, W, C]
        
        Returns:
            添加了2个坐标通道的特征图 [H, W, C+2] 或 [B, H, W, C+2]
        """
        # 创建归一化的网格坐标，范围[-1, 1]
        # meshgrid生成x和y坐标矩阵
        grid = jnp.array(
            np.stack(
                np.meshgrid(*[np.arange(s) / (s - 1) * 2 - 1 for s in x.shape[-3:-1]]),
                axis=-1,
            ),
            dtype=self.dtype,
        ).transpose(1, 0, 2)

        # 如果输入有batch维度，广播坐标网格到所有batch
        if x.ndim == 4:
            grid = jnp.broadcast_to(grid, [x.shape[0], *grid.shape])

        # 在通道维度拼接原始特征和坐标
        return jnp.concatenate([x, grid], axis=-1)


class SpatialSoftmax(nn.Module):
    """空间软最大值池化层
    
    将特征图转换为关键点表示，每个通道产生一个2D关键点坐标。
    通过softmax计算每个通道的空间注意力，然后计算期望坐标。
    
    这种方法在机器人视觉中很有用，可以提取类似关键点的表示。
    
    Attributes:
        height: 特征图高度
        width: 特征图宽度  
        channel: 特征图通道数
        pos_x: x坐标位置数组
        pos_y: y坐标位置数组
        temperature: softmax温度参数，-1表示学习温度
        log_heatmap: 是否记录热图（用于可视化）
    """
    height: int
    width: int
    channel: int
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    temperature: None
    log_heatmap: bool = False

    @nn.compact
    def __call__(self, features):
        """计算空间软最大值池化
        
        Args:
            features: 输入特征图 [H, W, C] 或 [B, H, W, C]
            
        Returns:
            关键点坐标 [2*C] 或 [B, 2*C]，每个通道对应一个(x,y)坐标
        """
        # 处理温度参数
        if self.temperature == -1:
            # 学习温度参数
            from jax.nn import initializers
            temperature = self.param(
                "softmax_temperature", initializers.ones, (1), jnp.float32
            )
        else:
            # 使用固定温度
            temperature = 1.0

        # 如果缺少batch维度则添加
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        assert len(features.shape) == 4
        batch_size, num_featuremaps = features.shape[0], features.shape[3]
        
        # 重塑特征图: [B, H, W, C] -> [B, C, H*W]
        features = features.transpose(0, 3, 1, 2).reshape(
            batch_size, num_featuremaps, self.height * self.width
        )

        # 对每个通道的空间位置应用softmax，得到注意力权重
        softmax_attention = nn.softmax(features / temperature)
        
        # 计算期望的x坐标：加权平均
        expected_x = jnp.sum(
            self.pos_x * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        
        # 计算期望的y坐标：加权平均
        expected_y = jnp.sum(
            self.pos_y * softmax_attention, axis=2, keepdims=True
        ).reshape(batch_size, num_featuremaps)
        
        # 拼接x和y坐标
        expected_xy = jnp.concatenate([expected_x, expected_y], axis=1)
        expected_xy = jnp.reshape(expected_xy, [batch_size, 2 * num_featuremaps])

        # 如果输入没有batch维度，移除添加的维度
        if no_batch_dim:
            expected_xy = expected_xy[0]
        return expected_xy


class SpatialLearnedEmbeddings(nn.Module):
    """空间学习嵌入层（ResNet版本）
    
    通过学习的空间权重将特征图聚合为固定维度向量。
    与spatial.py中的实现类似，但支持自定义初始化器。
    
    Attributes:
        height: 特征图高度
        width: 特征图宽度
        channel: 特征图通道数
        num_features: 输出特征数量
        kernel_init: 卷积核初始化器
        param_dtype: 参数数据类型
    """
    height: int
    width: int
    channel: int
    num_features: int = 5
    kernel_init: Callable = nn.initializers.lecun_normal()
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, features):
        """应用空间学习嵌入
        
        Args:
            features: 输入特征图 [B, H, W, C]
            
        Returns:
            嵌入向量 [B, num_features*channel]
        """
        # 初始化学习的空间权重
        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.height, self.width, self.channel, self.num_features),
            self.param_dtype,
        )

        # 如果缺少batch维度则添加
        no_batch_dim = len(features.shape) < 4
        if no_batch_dim:
            features = features[None]

        batch_size = features.shape[0]
        assert len(features.shape) == 4
        
        # 执行加权求和聚合
        features = jnp.sum(
            jnp.expand_dims(features, -1) * jnp.expand_dims(kernel, 0), axis=(1, 2)
        )
        # 展平为一维向量
        features = jnp.reshape(features, [batch_size, -1])

        # 移除添加的batch维度
        if no_batch_dim:
            features = features[0]

        return features


class MyGroupNorm(nn.GroupNorm):
    """自定义GroupNorm，支持3D输入
    
    扩展标准GroupNorm以处理没有batch维度的输入。
    如果输入是3D张量，自动添加和移除batch维度。
    """
    def __call__(self, x):
        """应用组归一化
        
        Args:
            x: 输入张量 [H, W, C] 或 [B, H, W, C]
            
        Returns:
            归一化后的张量，与输入形状相同
        """
        if x.ndim == 3:
            # 添加batch维度进行归一化
            x = x[jnp.newaxis]
            x = super().__call__(x)
            # 移除batch维度
            return x[0]
        else:
            return super().__call__(x)


class ResNetBlock(nn.Module):
    """标准ResNet残差块
    
    实现基本的ResNet块结构：
    x -> Conv -> Norm -> Act -> Conv -> Norm -> (+) -> Act
    |_______________________________________________|  (残差连接)
    
    Attributes:
        filters: 卷积核数量（输出通道数）
        conv: 卷积层模块
        norm: 归一化层模块
        act: 激活函数
        strides: 卷积步长，用于下采样
    """

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        """前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            残差块输出
        """
        # 保存残差连接
        residual = x
        
        # 主路径：第一个卷积块
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        
        # 主路径：第二个卷积块
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm()(y)

        # 如果形状不匹配，需要投影残差连接
        # 通常发生在通道数变化或空间下采样时
        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name="conv_proj")(
                residual
            )
            residual = self.norm(name="norm_proj")(residual)

        # 残差连接 + 激活
        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """瓶颈ResNet残差块
    
    使用1x1卷积降维和升维，中间使用3x3卷积，减少计算量。
    结构: 1x1(reduce) -> 3x3 -> 1x1(expand)
    输出通道数是输入的4倍。
    
    Attributes:
        filters: 中间层的卷积核数量（最终输出为4*filters）
        conv: 卷积层模块
        norm: 归一化层模块
        act: 激活函数
        strides: 卷积步长
    """

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        """前向传播
        
        Args:
            x: 输入特征图
            
        Returns:
            瓶颈块输出
        """
        residual = x
        
        # 1x1卷积降维
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        
        # 3x3卷积（可能包含下采样）
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        
        # 1x1卷积升维至4倍通道
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)  # scale初始为0，有助于训练

        # 如果形状不匹配，投影残差连接
        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name="conv_proj"
            )(residual)
            residual = self.norm(name="norm_proj")(residual)

        return self.act(residual + y)


class ResNetEncoder(nn.Module):
    """通用ResNet V1编码器
    
    支持多种配置选项的灵活ResNet实现，适用于机器人视觉任务。
    
    主要功能：
    - 多种池化方法：平均、最大值、空间softmax等
    - 支持FiLM条件化
    - 可添加空间坐标
    - 灵活的归一化选项
    
    Attributes:
        stage_sizes: 每个阶段的块数量 (e.g., (2,2,2,2) for ResNet-18)
        block_cls: 残差块类型 (ResNetBlock 或 BottleneckResNetBlock)
        num_filters: 初始卷积核数量
        dtype: 数据类型
        act: 激活函数名称
        conv: 卷积层类型
        norm: 归一化类型 ("group", "layer")
        add_spatial_coordinates: 是否添加空间坐标
        pooling_method: 池化方法 ("avg", "max", "spatial_softmax", "none"等)
        use_spatial_softmax: 是否使用空间softmax
        softmax_temperature: softmax温度参数
        use_multiplicative_cond: 是否使用乘法条件化
        num_spatial_blocks: 空间块数量
        use_film: 是否使用FiLM条件化
        bottleneck_dim: 瓶颈层维度（可选）
        pre_pooling: 是否在池化前返回（用于冻结编码器）
        image_size: 输入图像大小
    """

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    norm: str = "group"
    add_spatial_coordinates: bool = False
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    use_multiplicative_cond: bool = False
    num_spatial_blocks: int = 8
    use_film: bool = False
    bottleneck_dim: Optional[int] = None
    pre_pooling: bool = True
    image_size: tuple = (128, 128)

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        train: bool = True,
        cond_var=None,
        stop_gradient=False,
    ):
        """编码观察图像
        
        Args:
            observations: 输入图像 [B, H, W, C]，值范围[0, 255]
            train: 是否处于训练模式
            cond_var: 条件变量（用于FiLM或乘法条件化）
            stop_gradient: 是否停止梯度
            
        Returns:
            编码特征向量或特征图
        """
        # 图像预处理：调整大小
        if observations.shape[-3:-1] != self.image_size:
            observations = resize(observations, self.image_size)

        # 图像归一化：使用ImageNet的均值和标准差
        mean = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        # 可选：添加空间坐标通道
        if self.add_spatial_coordinates:
            x = AddSpatialCoordinates(dtype=self.dtype)(x)

        # 配置卷积层：不使用偏置，使用Kaiming初始化
        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        
        # 配置归一化层
        if self.norm == "batch":
            raise NotImplementedError
        elif self.norm == "group":
            # 组归一化：将通道分为4组
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            # 层归一化
            norm = partial(
                nn.LayerNorm,
                epsilon=1e-5,
                dtype=self.dtype,
            )
        else:
            raise ValueError("未找到归一化方法")

        # 获取激活函数
        act = getattr(nn, self.act)

        # 初始卷积层: 7x7卷积 + 步长2
        x = conv(
            self.num_filters,
            (7, 7),
            (2, 2),
            padding=[(3, 3), (3, 3)],
            name="conv_init",
        )(x)

        x = norm(name="norm_init")(x)
        x = act(x)
        # 最大池化层: 3x3核 + 步长2
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        # 构建ResNet阶段
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                # 每个阶段的第一个块使用步长2下采样（除了第一阶段）
                stride = (2, 2) if i > 0 and j == 0 else (1, 1)
                
                # 添加残差块，每个阶段通道数翻倍
                x = self.block_cls(
                    self.num_filters * 2**i,  # 第i阶段的通道数
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
                
                # 可选：应用FiLM条件化
                if self.use_film:
                    assert (
                        cond_var is not None
                    ), "条件变量为None，无法进行条件化"
                    x = FilmConditioning()(x, cond_var)
                    
                # 可选：应用乘法条件化
                if self.use_multiplicative_cond:
                    assert (
                        cond_var is not None
                    ), "条件变量为None，无法进行条件化"
                    # 将条件变量投影到特征通道数
                    cond_out = nn.Dense(
                        x.shape[-1], kernel_init=nn.initializers.xavier_normal()
                    )(cond_var)
                    # 扩展空间维度并乘以特征
                    x_mult = jnp.expand_dims(jnp.expand_dims(cond_out, 1), 1)
                    x = x * x_mult
        # 如果要求在池化前返回（用于冻结编码器）
        if self.pre_pooling:
            return jax.lax.stop_gradient(x)
            # return x

        # 应用池化层将空间特征图转换为向量
        if self.pooling_method == "spatial_learned_embeddings":
            # 使用学习的空间嵌入
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
            
        elif self.pooling_method == "spatial_softmax":
            # 使用空间softmax池化
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
            
        elif self.pooling_method == "avg":
            # 全局平均池化
            x = jnp.mean(x, axis=(-3, -2))
            
        elif self.pooling_method == "max":
            # 全局最大池化
            x = jnp.max(x, axis=(-3, -2))
            
        elif self.pooling_method == "none":
            # 不进行池化，保持空间结构
            pass
        else:
            raise ValueError("未找到池化方法")

        # 可选：添加瓶颈层
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x


class PreTrainedResNetEncoder(nn.Module):
    """预训练ResNet编码器封装
    
    用于封装预训练的ResNet编码器，并添加自定义的池化层。
    可以选择是否使用预训练编码器进行编码。
    
    Attributes:
        pooling_method: 池化方法
        use_spatial_softmax: 是否使用空间softmax
        softmax_temperature: softmax温度
        num_spatial_blocks: 空间块数量
        bottleneck_dim: 瓶颈层维度
        pretrained_encoder: 预训练的编码器模块
    """
    pooling_method: str = "avg"
    use_spatial_softmax: bool = False
    softmax_temperature: float = 1.0
    num_spatial_blocks: int = 8
    bottleneck_dim: Optional[int] = None
    pretrained_encoder: nn.module = None

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        encode: bool = True,
        train: bool = True,
    ):
        """前向传播
        
        Args:
            observations: 输入观察
            encode: 是否使用预训练编码器
            train: 是否处于训练模式
            
        Returns:
            编码特征
        """
        x = observations
        
        # 可选：使用预训练编码器进行编码
        if encode:
            x = self.pretrained_encoder(x, train=train)

        # 应用池化层（与ResNetEncoder中相同）
        if self.pooling_method == "spatial_learned_embeddings":
            height, width, channel = x.shape[-3:]
            x = SpatialLearnedEmbeddings(
                height=height,
                width=width,
                channel=channel,
                num_features=self.num_spatial_blocks,
            )(x)
            x = nn.Dropout(0.1, deterministic=not train)(x)
        elif self.pooling_method == "spatial_softmax":
            height, width, channel = x.shape[-3:]
            pos_x, pos_y = jnp.meshgrid(
                jnp.linspace(-1.0, 1.0, height), jnp.linspace(-1.0, 1.0, width)
            )
            pos_x = pos_x.reshape(height * width)
            pos_y = pos_y.reshape(height * width)
            x = SpatialSoftmax(
                height, width, channel, pos_x, pos_y, self.softmax_temperature
            )(x)
        elif self.pooling_method == "avg":
            x = jnp.mean(x, axis=(-3, -2))
        elif self.pooling_method == "max":
            x = jnp.max(x, axis=(-3, -2))
        elif self.pooling_method == "none":
            pass
        else:
            raise ValueError("未找到池化方法")

        # 可选：添加瓶颈层
        if self.bottleneck_dim is not None:
            x = nn.Dense(self.bottleneck_dim)(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        return x


# ResNet V1配置字典
# 定义了各种预设的ResNet架构配置
resnetv1_configs = {
    # ResNet-10: 4个阶段，每个阶段1个块
    "resnetv1-10": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock
    ),
    # ResNet-10冻结版本：在池化前返回特征图
    "resnetv1-10-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True
    ),
    # ResNet-18: 4个阶段，块数为(2,2,2,2)
    "resnetv1-18": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock
    ),
    # ResNet-18冻结版本
    "resnetv1-18-frozen": ft.partial(
        ResNetEncoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetBlock, pre_pooling=True
    ),
    # ResNet-34: 更深的网络
    "resnetv1-34": ft.partial(
        ResNetEncoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetBlock
    ),
    # ResNet-50: 使用瓶颈结构
    "resnetv1-50": ft.partial(
        ResNetEncoder, stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock
    ),
    # ResNet-18更深版本: (3,3,3,3)
    "resnetv1-18-deeper": ft.partial(
        ResNetEncoder, stage_sizes=(3, 3, 3, 3), block_cls=ResNetBlock
    ),
    # ResNet-18最深版本: (4,4,4,4)
    "resnetv1-18-deepest": ft.partial(
        ResNetEncoder, stage_sizes=(4, 4, 4, 4), block_cls=ResNetBlock
    ),
    # ResNet-18 Bridge版本: 带空间学习嵌入
    "resnetv1-18-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(2, 2, 2, 2),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    # ResNet-34 Bridge版本
    "resnetv1-34-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
    ),
    # ResNet-34 Bridge + FiLM条件化
    "resnetv1-34-bridge-film": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=ResNetBlock,
        num_spatial_blocks=8,
        use_film=True,
    ),
    # ResNet-50 Bridge版本
    "resnetv1-50-bridge": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
    ),
    # ResNet-50 Bridge + FiLM条件化
    "resnetv1-50-bridge-film": ft.partial(
        ResNetEncoder,
        stage_sizes=(3, 4, 6, 3),
        block_cls=BottleneckResNetBlock,
        num_spatial_blocks=8,
        use_film=True,
    ),
}
