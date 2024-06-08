import getopt
import sys
import typing
import os

import numpy as np
import tensorflow as tf
# from gym.wrappers import Monitor
from PIL import Image
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from constants import (CONST_ENV_NAME, BATCH_SIZE, REPLAY_CAPACITY, TIME_STEPS_PER_SAVE,
                    TIME_STEPS_PER_TRAIN, TRAIN_STEPS_PER_Q_SYNC, 
                    NUM_TIME_STEPS, SAVES_PATH, PLAY_ROUNDS)
from dqn import DQN, LayerConfig, run_dqn_algorithm
from gym_interface import EnvProtocol, gym
from results import plot_records

ImageLike = typing.Union[np.ndarray, Image.Image, tf.Tensor]


class AirRaidPreprocessor:
    env_obsv_shape = (160, 250, 3)  # shape=(width, height, n_channels)
    preprocessed_obsv_shape = (80, 80, 1)  # shape=(width, height)

    def __call__(self, obsv: ImageLike) -> np.ndarray:
        if isinstance(obsv, Image.Image):
            im = obsv
        elif isinstance(obsv, (np.ndarray, tf.Tensor)):
            obsv = tf.convert_to_tensor(obsv)
            obsv = np.asarray(obsv)
            im = Image.fromarray(obsv)
        else:
            raise ValueError(
                f"Unsupported state type: `{type(obsv)}`. "
                f"Only `PIL.Image.Image`, `np.ndarray` and `tf.Tensor` "
                f"are supported."
            )

        im = im.convert("L")
        im.thumbnail((80, 125), Image.Resampling.LANCZOS)  # shape=(80, 125)

        width, height = im.size
        # Crop the image according to the (l, r, u, d) borders:
        im = im.crop((0, height - 80, width, height))  # shape=(80, 80)

        p_obsv = np.asarray(im) / 255.0  # shape=(80, 80)
        p_obsv = np.expand_dims(p_obsv, -1)  # shape=(80, 80, 1)

        return p_obsv


airraid_preprocessor = AirRaidPreprocessor()

airraid_dqn_layer_configs = (
    LayerConfig(Conv2D, dict(
        kernel_size=(8, 8), filters=16, strides=4,
        activation="relu",
        name="conv1"
    )),
    LayerConfig(Conv2D, dict(
        kernel_size=(4, 4), filters=32, strides=1,
        activation="relu",
        name="conv2"
    )),
    LayerConfig(Flatten, {}),
    LayerConfig(Dense, dict(units=256, activation="relu", name="fc1")),
    LayerConfig(Dense, dict(units=gym.make(CONST_ENV_NAME).action_space.n,
                            activation=None, name="q_layer"))
)


def train_airraid(
    display=False,
    save_path=None
):
    if display:
        env = gym.make(CONST_ENV_NAME, render_mode="human")
    else:
        env: EnvProtocol = gym.make(CONST_ENV_NAME)

    print('TRAIN')
    run_dqn_algorithm(
        env,
        airraid_dqn_layer_configs,
        preprocess=airraid_preprocessor,
        preprocessed_obsv_shape=airraid_preprocessor.preprocessed_obsv_shape,
        load_path=None,#save_path,
        save_path=save_path,
        num_time_steps=NUM_TIME_STEPS
    )
    env.close()


def play_airraid(
    play_rounds=100,
    display=False,
    load_save_path=SAVES_PATH
):
    if display:
        env = gym.make(CONST_ENV_NAME, render_mode="human")
    else:
        env = gym.make(CONST_ENV_NAME)
    dqn: DQN = DQN(
        state_shape=(1, *airraid_preprocessor.preprocessed_obsv_shape),
        layer_configs=airraid_dqn_layer_configs
    )
    dqn.load(load_save_path)

    total_reward = 0.0
    for i in range(play_rounds):
        observation = env.reset()[0]
        observation = airraid_preprocessor(observation)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = dqn.select_action(np.array([observation]))
            observation, reward, terminated, truncated, _ = env.step(action)
            observation = airraid_preprocessor(observation)
            total_reward += reward

    aver_reward = total_reward / play_rounds
    print('Average reward in {0} rounds of game: {1}'
          .format(play_rounds, aver_reward))

    env.close()


if __name__ == '__main__':
    mode = ''
    try:
        ovpairs, args = getopt.getopt(sys.argv[1:], "hsm:", ["help", "mode", "stat"])
    except getopt.GetoptError:
        print("Invalid argument!")
        sys.exit(1)
    else:
        if len(ovpairs) != 0:
            for opt, val in ovpairs:
                if opt in ("-h", "--help"):
                    print("Usage: python airraid.py [-h|-m t|-m p|-s]")
                    sys.exit()
                elif opt in ("-m", "--mode"):
                    mode = val
                elif opt in ("-s", "--stat"):
                    plot_records(record_dir=SAVES_PATH, save=True)
                    sys.exit()
        else:
            mode = 't'

    if mode in ("t", "train"):
        train_airraid(display=False, save_path=SAVES_PATH)
    elif mode in ("p", "play"):
        play_airraid(play_rounds=PLAY_ROUNDS, display=True,
                     load_save_path=SAVES_PATH)
    else:
        print("Invalid argument! Mode can only be 't' or 'p', but '{0}' "
              "received.".format(mode))