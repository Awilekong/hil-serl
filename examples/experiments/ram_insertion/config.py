import os
import sys
import jax
import jax.numpy as jnp
import numpy as np

# Add serl_robot_infra and serl_launcher to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.insert(0, os.path.join(project_root, 'serl_robot_infra'))
sys.path.insert(0, os.path.join(project_root, 'serl_launcher'))

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.ram_insertion.wrapper import RAMEnv

class EnvConfig(DefaultEnvConfig):
    SERVER_URL = "http://192.168.31.1:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "323622271399",
            "dim": (1280, 720),
            "exposure": 3000,
        },
        "wrist_2": {
            "serial_number": "323622271298",
            "dim": (1280, 720),
            "exposure": 3000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[3:697, 164:1100],
        "wrist_2": lambda img: img[1:710, 225:1246],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.08, 0.06, 0.03, 0.03, 0.3, 0.8])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.08, 0.06, 0.12, 0.03, 0.3, 0.8])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)  # Original scale for training
    DISPLAY_IMAGE = True
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.02,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.015,
        "translational_clip_neg_x": 0.006,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.012,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 30
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        print("[DEBUG] 步骤 1/7: 创建 RAMEnv...")
        env = RAMEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        print("[DEBUG] 步骤 1/7: RAMEnv 创建完成")
        
        env = GripperCloseEnv(env)  # Commented out to allow gripper control
        
        if not fake_env:
            print("[DEBUG] 步骤 2/7: 添加 SpacemouseIntervention...")
            env = SpacemouseIntervention(env)
            print("[DEBUG] 步骤 2/7: SpacemouseIntervention 完成")
        else:
            print("[DEBUG] 步骤 2/7: 跳过 (fake_env=True)")
            
        print("[DEBUG] 步骤 3/7: 添加 RelativeFrame...")
        env = RelativeFrame(env)
        print("[DEBUG] 步骤 4/7: 添加 Quat2EulerWrapper...")
        env = Quat2EulerWrapper(env)
        print("[DEBUG] 步骤 5/7: 添加 SERLObsWrapper...")
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        print("[DEBUG] 步骤 6/7: 添加 ChunkingWrapper...")
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                # classifier output shape: (batch_size=1, 1) due to ChunkingWrapper
                classifier_logits = classifier(obs)  # shape: (1, 1)
                # Squeeze to get scalar
                classifier_logits = jnp.squeeze(classifier_logits)  # shape: ()
                classifier_prob = sigmoid(classifier_logits)
                
                # added check for z position to further robustify classifier
                # obs['state'] has shape (1, state_dim) due to ChunkingWrapper
                z_position = obs['state'][0, 6]
                
                # Return 1 if both conditions met, 0 otherwise
                # reward = (classifier_prob > 0.85) & (z_position > 0.04)
                reward = classifier_prob > 0.5

                return int(reward)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env