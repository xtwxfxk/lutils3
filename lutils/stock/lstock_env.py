import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_NUM_AMOUNTS = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 240 # 40000

INITIAL_ACCOUNT_BALANCE = 10000


class LStockDaily1MinEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(LStockDaily1MinEnv, self).__init__()

        self.days = []
        self.df = df
        row_index = 0
        while row_index < df.shape[0]:
            self.days.append(df[row_index:row_index+240])
            row_index = row_index + 240
        if self.days[-1].shape[0] < 240:
            self.days.pop(-1)

        self.day = random.choice(self.days)
        self.current_step = 0

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # self.reward_range = (0, 1)
        # self.reward_range = (0, 100)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

        self.balance = INITIAL_ACCOUNT_BALANCE

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.day.iloc[self.current_step: self.current_step + 6]['open'].values / MAX_SHARE_PRICE,
            self.day.iloc[self.current_step: self.current_step + 6]['high'].values / MAX_SHARE_PRICE,
            self.day.iloc[self.current_step: self.current_step + 6]['low'].values / MAX_SHARE_PRICE,
            self.day.iloc[self.current_step: self.current_step + 6]['close'].values / MAX_SHARE_PRICE,
            self.day.iloc[self.current_step: self.current_step + 6]['vol'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.day.iloc[self.current_step + 6]['close'] + 0.02

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step = self.current_step + 1

        delay_modifier = (self.current_step / MAX_STEPS)

        # reward = self.balance - INITIAL_ACCOUNT_BALANCE
        # reward = self.balance * delay_modifier
        # done = self.net_worth <= 0
        done = self.current_step > (self.day.shape[0]-7) or self.net_worth <= 0
        # reward = 1 if self.net_worth > self.balance else 0
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        if reward < 0: reward = 0
        # if done:
        #     # reward = 1 if self.net_worth > self.balance else 0
        #     self.balance = self.net_worth
        # else:
        #     reward = 0


        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        self.day = random.choice(self.days)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')



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
    model.learn(total_timesteps=40000)

    obs = env.reset()

    rewards = []
    for i in range(220):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        env.render()


    plt.plot(rewards)
    plt.show()