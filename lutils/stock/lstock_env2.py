import os, subprocess, time, signal
import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

from gym import error, spaces
from gym import utils
from gym.utils import seeding


MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_NUM_AMOUNTS = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 240 # 40000

INITIAL_ACCOUNT_BALANCE = 10000


class LStockDailyEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = None

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        self.observation_space = spaces.Box(low=0, high=1, shape=(6, 6), dtype=np.float16)

        self.balance = INITIAL_ACCOUNT_BALANCE


    def _step(self, action):
        self._take_action(action)
        self.status = self.env.step()
        reward = self._get_reward()
        ob = self.env.getState()
        episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, episode_over, {}

    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.DASH:
            self.env.act(action_type, action[1], action[2])
        elif action_type == hfo_py.TURN:
            self.env.act(action_type, action[3])
        elif action_type == hfo_py.KICK:
            self.env.act(action_type, action[4], action[5])
        else:
            print('Unrecognized action %d' % action_type)
            self.env.act(hfo_py.NOOP)

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def _reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        while self.status != hfo_py.IN_GAME:
            self.env.act(hfo_py.NOOP)
            self.status = self.env.step()
        return self.env.getState()

        

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')


ACTION_LOOKUP = {
    0 : hfo_py.DASH,
    1 : hfo_py.TURN,
    2 : hfo_py.KICK,
    3 : hfo_py.TACKLE, # Used on defense to slide tackle the ball
    4 : hfo_py.CATCH,  # Used only by goalie to catch the ball
}



if __name__ == '__main__':
    import gym
    import json
    import datetime as dt
    import matplotlib.pyplot as plt

    from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, ActorCriticPolicy, LstmPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import PPO2, PPO1, A2C, DQN, TD3, SAC

    import pandas as pd

    from lutils.stock import LTdxHq

    ltdxhq = LTdxHq()
    df = ltdxhq.get_k_data_1min('603636')
    # df = ltdxhq.get_k_data_5min('603636')
    # df = ltdxhq.get_k_data_daily('603636')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: LStockDaily1MinEnv(df)])

    model = PPO2(MlpPolicy, env, verbose=1)
    # model = PPO1(LstmPolicy, env, verbose=1)
    model.learn(total_timesteps=100000)

    obs = env.reset()

    rewards = []
    for i in range(220):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        env.render()


    plt.plot(rewards)
    plt.show()