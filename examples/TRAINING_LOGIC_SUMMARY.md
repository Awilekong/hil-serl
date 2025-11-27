# SERL训练逻辑详细总结

## 📋 目录
1. [整体架构](#整体架构)
2. [训练流程](#训练流程)
3. [策略网络](#策略网络)
4. [奖励与Reset逻辑](#奖励与reset逻辑)
5. [数据流](#数据流)
6. [关键参数](#关键参数)

---

## 🏗️ 整体架构

### 双进程架构：Learner + Actor

```
┌─────────────────────────────────────────────────────────────┐
│                      训练系统架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         网络参数          ┌──────────┐ │
│  │   Learner进程    │ ◄─────────────────────── │  Actor   │ │
│  │  (run_learner)   │                          │ 进程     │ │
│  │                  │                          │(run_actor)│ │
│  │ - 不连机器人      │                          │          │ │
│  │ - GPU训练        │ ──────────────────────► │ - 连机器人│ │
│  │ - 更新网络        │      发送新策略参数        │ - CPU执行│ │
│  └────────┬─────────┘                          └────┬─────┘ │
│           │                                         │       │
│           │                                         │       │
│      读取Demo                                   采集新数据   │
│           │                                         │       │
│           ▼                                         ▼       │
│  ┌──────────────────┐                    ┌──────────────┐  │
│  │  Demo Buffer     │                    │ Replay Buffer│  │
│  │  (50%数据)       │                    │  (50%数据)   │  │
│  └──────────────────┘                    └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 训练流程

### 1. Learner进程 (`run_learner.sh`)

**启动命令：**
```bash
python train_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=charge_first_run \
    --demo_path=<demo1.pkl> \
    --demo_path=<demo2.pkl> \
    --learner
```

**主要职责：**

```python
# 1️⃣ 初始化
config = CONFIG_MAPPING["ram_insertion"]()  # 加载配置
env = config.get_environment(
    fake_env=True,        # ← 关键！不连接真实机器人
    classifier=True       # 加载reward classifier
)

# 2️⃣ 加载Demo数据到demo_buffer
demo_buffer = load_demos(FLAGS.demo_path)

# 3️⃣ 创建TrainerServer，等待actor连接
server = TrainerServer()
server.register_data_store("actor_env", replay_buffer)
server.register_data_store("actor_env_intvn", demo_buffer)
server.start()

# 4️⃣ 等待replay_buffer填充到training_starts（默认5000条transitions）
while len(replay_buffer) < config.training_starts:
    time.sleep(1)

# 5️⃣ 发送初始网络参数给actor
server.publish_network(agent.state.params)

# 6️⃣ 训练循环 (max_steps次迭代)
for step in range(config.max_steps):
    # 50/50采样：一半来自demo，一半来自在线数据
    batch_online = next(replay_buffer.iterator)      # 50% 在线数据
    batch_demo = next(demo_buffer.iterator)          # 50% Demo数据
    batch = concat([batch_online, batch_demo])
    
    # 更新策略网络
    agent, info = agent.update(
        batch,
        networks_to_update=["critic", "actor", "temperature"]
    )
    
    # 每steps_per_update步发送新参数给actor
    if step % config.steps_per_update == 0:
        server.publish_network(agent.state.params)
    
    # 每checkpoint_period步保存checkpoint
    if step % config.checkpoint_period == 0:
        save_checkpoint(agent.state, step)
```

---

### 2. Actor进程 (`run_actor.sh`)

**启动命令：**
```bash
python train_rlpd.py \
    --exp_name=ram_insertion \
    --checkpoint_path=charge_first_run \
    --actor
```

**主要职责：**

```python
# 1️⃣ 初始化
env = config.get_environment(
    fake_env=False,       # ← 关键！连接真实机器人
    classifier=True       # 加载reward classifier用于计算reward
)

# 2️⃣ 连接到learner
client = TrainerClient(
    "actor_env",
    FLAGS.ip,  # learner的IP地址
    data_stores={"actor_env": data_store, "actor_env_intvn": intvn_data_store}
)

# 3️⃣ 注册网络参数更新回调
def update_params(params):
    agent = agent.replace(state=agent.state.replace(params=params))

client.recv_network_callback(update_params)

# 4️⃣ 探索循环 (max_steps次)
obs, _ = env.reset()
for step in range(config.max_steps):
    # 前random_steps步随机探索
    if step < config.random_steps:
        actions = env.action_space.sample()
    else:
        # 使用策略网络采样动作
        actions = agent.sample_actions(
            observations=obs,
            argmax=False,  # 随机采样，用于探索
            seed=rng_key
        )
    
    # 执行动作
    next_obs, reward, done, truncated, info = env.step(actions)
    
    # 检查人类干预
    if "intervene_action" in info:
        actions = info["intervene_action"]  # 使用人类动作
        intvn_data_store.insert(transition)  # 存入demo buffer
    
    # 存储transition
    transition = {
        "observations": obs,
        "actions": actions,
        "next_observations": next_obs,
        "rewards": reward,
        "masks": 1.0 - done,
        "dones": done
    }
    data_store.insert(transition)
    
    obs = next_obs
    
    # ⭐ 关键！如果done或truncated，reset环境
    if done or truncated:
        client.update()  # 发送统计信息给learner
        obs, _ = env.reset()  # ← 重置环境，开始新episode
```

---

## 🎯 策略网络 (Policy Network)

### 输入 (Observations)

策略网络接收的观察包含：

```python
observations = {
    # 图像输入 (根据config.image_keys)
    "wrist_1": np.array(shape=(1, H, W, 3)),      # 手腕相机1
    "wrist_2": np.array(shape=(1, H, W, 3)),      # 手腕相机2
    "side_policy": np.array(shape=(1, H, W, 3)),  # 侧面相机
    
    # 状态输入 (根据config.proprio_keys)
    "state": np.array(shape=(1, state_dim)),
    # state包含: [tcp_pose(6), tcp_force(3), tcp_torque(3)] = 12维
    # 详细: [x, y, z, roll, pitch, yaw, fx, fy, fz, tx, ty, tz]
}
```

**关键说明：**
- 所有输入都有batch维度 `(1, ...)`，因为ChunkingWrapper添加了时间维度
- 图像经过裁剪（IMAGE_CROP）后输入
- 状态是归一化后的本体感受信息

### 输出 (Actions)

```python
actions = agent.sample_actions(observations, seed=rng_key, argmax=False)
# 输出形状: (action_dim,)

# 对于 setup_mode="single-arm-fixed-gripper":
actions.shape = (6,)  # [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]

# 对于 setup_mode="single-arm-learned-gripper":
actions.shape = (7,)  # [...上面6个..., gripper_command]
```

**动作空间说明：**
- 动作是**相对增量**，不是绝对位置
- 经过ACTION_SCALE缩放：`(0.01, 0.06, 1)` → 位置±1cm, 旋转±3.4°
- RelativeFrame包装器将相对动作转换为绝对目标位置
- 然后发送给Franka机器人的impedance controller

### 策略网络架构

```python
# 视觉编码器 (encoder_type="resnet-pretrained")
images → ResNet18 (预训练) → 特征向量 (512维)

# 状态编码器
state → MLP(256) → 特征向量 (256维)

# 融合与策略头
concat([image_features, state_features]) → MLP(1024, 1024) → 
    → mean (action_dim)
    → log_std (action_dim)
    
# 输出分布
TanhNormal(mean, std) → 采样动作 → tanh压缩到[-1, 1]
```

---

## 🎁 奖励与Reset逻辑

### 奖励计算

**来源：Reward Classifier**

```python
# 在config.py的get_environment()中
classifier = load_classifier_func(
    checkpoint_path="./classifier_ckpt/",
    image_keys=["side_policy"]  # 只用side_policy相机
)

def reward_func(obs):
    # 使用训练好的分类器判断成功
    logits = classifier(obs)
    prob = sigmoid(logits)
    
    # 阈值判断
    reward = 1 if prob > 0.5 else 0
    return reward

env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
```

**奖励特性：**
- **稀疏奖励**：只有 `0` 或 `1` 两种值
- `reward=1`：任务成功（如RAM插入成功）
- `reward=0`：任务进行中或失败

### Episode终止条件 (Done)

```python
# 在franka_env.py的step()方法中
done = (
    self.curr_path_length >= self.max_episode_length  # 超时 (默认100步)
    or reward == 1                                     # 任务成功
    or self.terminate                                  # 手动终止
)
```

**三种终止情况：**

1. **超时终止** (`MAX_EPISODE_LENGTH=100`)
   - Episode达到100步还未成功
   - `done=True`, `reward=0`

2. **成功终止** (Classifier判定为成功)
   - Classifier输出`reward=1`
   - `done=True`, `reward=1`
   - 🎉 **成功案例！**

3. **手动终止** (紧急停止)
   - 按下紧急停止按钮
   - `done=True`, `reward=0`

### Reset逻辑

```python
# 在actor循环中
if done or truncated:
    # 📊 发送episode统计信息
    stats = {
        "episode_return": running_return,
        "episode_length": curr_path_length,
        "intervention_count": intervention_count,
        "succeed": reward
    }
    client.request("send-stats", stats)
    
    # 🔄 重置环境
    obs, _ = env.reset()
    
    # Reset会：
    # 1. 移动机器人到RESET_POSE (TARGET_POSE + [0, 0, 0.05, 0, 0.05, 0])
    # 2. 如果RANDOM_RESET=True，添加随机扰动:
    #    - XY: ±RANDOM_XY_RANGE (0.01m)
    #    - RZ: ±RANDOM_RZ_RANGE (0.01rad)
    # 3. 重置episode计数器
    # 4. 获取新的初始观察
```

**关键点：**
- ✅ **是的！reward=1时会立即reset**
- Reset后机器人回到起始位置附近
- 每个episode独立，成功后不会继续在成功状态下探索

---

## 📊 数据流

### Transition格式

```python
transition = {
    "observations": {
        "wrist_1": image_array,
        "wrist_2": image_array, 
        "side_policy": image_array,
        "state": state_vector
    },
    "actions": action_array,           # (6,) 或 (7,)
    "next_observations": {...},        # 同observations
    "rewards": float,                  # 0 或 1
    "masks": float,                    # 1.0 - done
    "dones": bool,                     # True/False
}
```

### 数据存储

**两个Buffer：**

1. **Replay Buffer** (在线数据)
   - 存储actor采集的所有transitions
   - 包括策略探索的数据
   - 包括人类干预的数据

2. **Demo Buffer** (Demo数据)
   - 存储预先录制的demo数据（从pkl文件加载）
   - 存储actor运行时人类干预的数据

### RLPD采样策略

```python
# 50/50采样比例
batch = {
    50% from replay_buffer,  # 在线探索数据
    50% from demo_buffer     # 高质量demo数据
}

# 为什么这样设计？
# 1. Demo数据提供高质量的成功轨迹
# 2. 在线数据提供探索和分布覆盖
# 3. 结合两者加速学习和提高稳定性
```

---

## ⚙️ 关键参数

### 训练参数 (TrainConfig)

```python
class TrainConfig:
    # 网络架构
    encoder_type = "resnet-pretrained"      # 视觉编码器
    setup_mode = "single-arm-fixed-gripper" # 动作空间配置
    
    # 数据配置
    image_keys = ["wrist_1", "wrist_2", "side_policy"]
    classifier_keys = ["side_policy"]       # Classifier用哪个相机
    proprio_keys = ["tcp_pose", "tcp_force", "tcp_torque"]
    
    # 训练超参数
    batch_size = 256                        # 训练batch大小
    max_steps = 100000                      # 最大训练步数
    random_steps = 300                      # 前300步随机探索
    training_starts = 5000                  # 开始训练前需要的数据量
    
    # 更新频率
    steps_per_update = 50                   # 每50步发送新参数给actor
    checkpoint_period = 50                  # 每50步保存checkpoint
    log_period = 100                        # 每100步记录日志
    buffer_period = 1000                    # 每1000步保存buffer到磁盘
    
    # 学习率
    cta_ratio = 2                           # Critic更新次数 / Actor更新次数
    discount = 0.99                         # 折扣因子
```

### 环境参数 (EnvConfig)

```python
class EnvConfig:
    # 位姿配置
    TARGET_POSE = [0.497, 0.092, 0.361, 3.102, 0.012, 0.172]
    RESET_POSE = TARGET_POSE + [0, 0, 0.05, 0, 0.05, 0]
    
    # 安全限制
    ABS_POSE_LIMIT_LOW = TARGET_POSE - [0.08, 0.06, 0.03, 0.03, 0.3, 0.8]
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + [0.08, 0.06, 0.12, 0.03, 0.3, 0.8]
    
    # 动作缩放
    ACTION_SCALE = (0.01, 0.06, 1)          # 位置±1cm, 旋转±3.4°
    
    # Episode配置
    MAX_EPISODE_LENGTH = 100                # 最大步数
    RANDOM_RESET = True                     # 是否随机reset
    RANDOM_XY_RANGE = 0.01                  # XY随机范围
    RANDOM_RZ_RANGE = 0.01                  # RZ随机范围
    
    # 相机配置
    REALSENSE_CAMERAS = {...}               # 相机序列号、分辨率、曝光
    IMAGE_CROP = {...}                      # 图像裁剪lambda函数
    
    # 阻抗控制参数
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "rotational_stiffness": 150,
        ...
    }
```

---

## 🎓 训练流程总结

### 完整循环

```
1️⃣ 启动Learner进程
   └─ 加载config和demo数据
   └─ 创建fake_env（不连机器人）
   └─ 等待replay_buffer填充

2️⃣ 启动Actor进程
   └─ 加载config和classifier
   └─ 创建真实env（连接机器人）
   └─ 连接到Learner

3️⃣ Actor探索循环 (每个episode):
   ├─ Reset机器人到起始位置
   ├─ 循环执行动作 (最多100步):
   │  ├─ 从策略网络采样动作
   │  ├─ 执行动作
   │  ├─ Classifier计算reward
   │  ├─ 存储transition
   │  └─ 如果reward=1或超时 → done=True → 跳出
   └─ Reset环境，开始新episode

4️⃣ Learner训练循环:
   ├─ 从replay_buffer和demo_buffer各采样50%
   ├─ 更新Critic网络 (2次)
   ├─ 更新Actor和Temperature (1次)
   ├─ 每50步发送新参数给Actor
   └─ 每50步保存checkpoint

5️⃣ 持续迭代直到max_steps
```

### 关键特性

- ✅ **异步训练**：Learner和Actor并行运行
- ✅ **稀疏奖励**：只有0/1，靠classifier判断成功
- ✅ **立即Reset**：成功后立即reset，不会继续探索成功状态
- ✅ **人类干预**：SpaceMouse干预的数据会特别标记并存入demo_buffer
- ✅ **50/50采样**：平衡demo数据和在线数据
- ✅ **相对动作**：策略输出相对增量，不是绝对位置

---

## 📝 常见问题

**Q: 为什么reward=1后要reset？**
A: 因为任务已经完成，继续在成功状态下探索没有意义。Reset后开始新的尝试，收集更多样化的数据。

**Q: 策略网络看到的是什么？**
A: 3个相机的RGB图像 + 12维状态向量（位置、力、力矩）

**Q: 动作空间是什么？**
A: 6D相对增量 [dx, dy, dz, droll, dpitch, dyaw]，经过ACTION_SCALE缩放后发送给阻抗控制器

**Q: Classifier在哪里训练？**
A: 使用`train_reward_classifier.py`单独训练，使用success/failure图像数据

**Q: 为什么需要两个进程？**
A: Learner用GPU高效训练，Actor用CPU与机器人交互。分离后互不阻塞，提高效率。

---

生成时间: 2025-11-27
文件位置: `/home/dexfranka/ws_zpw/hil-serl/examples/TRAINING_LOGIC_SUMMARY.md`
