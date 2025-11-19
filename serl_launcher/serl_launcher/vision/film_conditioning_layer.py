"""
FiLM (Feature-wise Linear Modulation) 条件化层

该模块实现了FiLM机制，用于将条件信息（如语言指令、状态等）融合到卷积特征图中。
FiLM通过学习每个通道的缩放(scale)和偏移(shift)参数来调制特征图。

改编自: https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py

参考文献:
- FiLM: Visual Reasoning with a General Conditioning Layer (Perez et al., 2018)
"""
import flax.linen as nn
import jax.numpy as jnp


class FilmConditioning(nn.Module):
    """FiLM条件化模块，用于将条件信息融合到卷积特征中"""
    @nn.compact
    def __call__(self, conv_filters: jnp.ndarray, conditioning: jnp.ndarray):
        """应用FiLM条件化到卷积特征图

        FiLM通过以下公式调制特征: output = conv_filters * (1 + gamma) + beta
        其中gamma和beta是从条件信息中学习得到的参数。

        Args:
            conv_filters: 卷积特征图，形状为 [batch_size, height, width, channels]
            conditioning: 条件信息向量，形状为 [batch_size, conditioning_size]
                         例如：语言嵌入、状态向量等

        Returns:
            调制后的特征图，形状为 [batch_size, height, width, channels]
        """
        # 将条件信息投影到特征通道维度，用于加法调制(beta参数)
        # 初始化为0，使得初始时FiLM不改变特征
        projected_cond_add = nn.Dense(
            features=conv_filters.shape[-1],  # 输出维度等于特征通道数
            kernel_init=nn.initializers.zeros,  # 权重初始化为0
            bias_init=nn.initializers.zeros,    # 偏置初始化为0
        )(conditioning)
        
        # 将条件信息投影到特征通道维度，用于乘法调制(gamma参数)
        projected_cond_mult = nn.Dense(
            features=conv_filters.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(conditioning)

        # 扩展维度以匹配空间维度 [batch_size, conditioning_size] -> [batch_size, 1, 1, channels]
        # 这样可以对每个空间位置应用相同的调制参数
        projected_cond_add = projected_cond_add[..., None, None, :]
        projected_cond_mult = projected_cond_mult[..., None, None, :]

        # 应用FiLM调制: output = input * (1 + gamma) + beta
        # gamma控制缩放，beta控制偏移
        return conv_filters * (1 + projected_cond_add) + projected_cond_mult

if __name__ == "__main__":
    """测试代码：验证FiLM层的功能"""
    import jax
    import jax.numpy as jnp

    # 初始化随机数生成器
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    # 创建示例卷积特征图: [batch_size=1, height=32, width=32, channels=3]
    x = jax.random.normal(subkey, (1, 32, 32, 3))
    x = jnp.array(x)

    # 创建示例条件向量: [batch_size=1, conditioning_size=64]
    z = jnp.ones((1, 64))
    
    # 初始化FiLM层
    film = FilmConditioning()
    
    # 初始化参数
    params = film.init(key, x, z)
    
    # 应用FiLM条件化
    y = film.apply(params, x, z)

    print(y.shape)  # 应该输出 (1, 32, 32, 3)
