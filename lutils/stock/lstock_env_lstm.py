import random, datetime
import gym
from gym import error, spaces
import pandas as pd
import numpy as np
from stockstats import StockDataFrame

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_NUM_AMOUNTS = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 60
MAX_STEPS = 240 # 40000
NEXT_OBSERVATION_SIZE = 5

INITIAL_ACCOUNT_BALANCE = 10000

# writer = SummaryWriter('log')


class LStockDailyEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(LStockDailyEnv, self).__init__()

        self.step_index = 1

        self.df = df

        self.current_step = NEXT_OBSERVATION_SIZE

        self.reward_range = (-np.inf, np.inf)

        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        self.observation_space = spaces.Box(low=-1, high=1, shape=(14, NEXT_OBSERVATION_SIZE), dtype=np.float16)


    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['open'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['high'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['low'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['close'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['vol'].values / MAX_NUM_SHARES,

            # self.df['macd'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['macdh'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['macds'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['volume_delta'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['open_2_d'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['open_-2_r'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['cr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['cr-ma1'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # # self.df['cr-ma2'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # # self.df['cr-ma3'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['kdjk'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['kdjd'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['kdjj'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['open_2_sma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['dma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['pdi'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['mdi'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['dx'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['adx'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['adxr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # # self.df['tema'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # # self.df['vr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # # self.df['vr_6_sma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,

            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['open'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['high'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['low'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['close'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['vol'].values / MAX_NUM_SHARES,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['amount'].values / MAX_NUM_SHARES,

            self.df['macd'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['macdh'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['macds'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['kdjk'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['kdjd'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['kdjj'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,

            self.df['rsi_6'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['rsi_12'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,

        ])

        return frame

    def _take_action(self, action):
        current_price = self.df.iloc[self.current_step]['close'] # + 0.02
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
        self.step_index = self.step_index + 1
        shares_held = self.shares_held
        self._take_action(action)

        self.current_step = self.current_step + 1

        done = self.net_worth <= INITIAL_ACCOUNT_BALANCE * .9 or datetime.datetime.strptime(self.df.index[self.current_step], '%Y-%M-%d').weekday() == 4

        obs = self._next_observation()

        reward = 0

        action_type = action[0]
        if action_type < 1: # Buy
            reward = self.df.iloc[self.current_step + 1]['close'] - self.df.iloc[self.current_step]['close']
        elif action_type >= 1 and action_type < 2: # Sell
            reward = self.df.iloc[self.current_step]['close'] - self.df.iloc[self.current_step + 1]['close']
        else: # Hold
            reward = self.df.iloc[self.current_step + 1]['close'] - self.df.iloc[self.current_step]['close']

        if done:
            reward = (self.net_worth - INITIAL_ACCOUNT_BALANCE) * 10

        return obs, reward, done, {'net_worth': self.net_worth}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        if self.current_step + 5 >= self.df.shape[0]:
            self.current_step = NEXT_OBSERVATION_SIZE
        # else:
            # self.current_step = self.current_step + 1

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



def test2():
    import gym
    import datetime as dt
    import matplotlib.pyplot as plt

    from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, ActorCriticPolicy, LstmPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.common.evaluation import evaluate_policy
    from stable_baselines import PPO2, PPO1, A2C, DQN, TD3, SAC

    import pandas as pd

    from lutils.stock import LTdxHq

    code = '603636' # 000032 300142 603636 600519
    ltdxhq = LTdxHq()
    # df = ltdxhq.get_k_data_1min('603636') # 000032 300142 603636 600519
    # df = ltdxhq.get_k_data_5min('603636')
    df = ltdxhq.get_k_data_daily(code, end='2020-01-01')
    eval_df = ltdxhq.get_k_data_daily(code, start='2020-01-01')
    ltdxhq.close()

    df = StockDataFrame(df) # .rename(columns={'vol': 'volume'}))
    env = DummyVecEnv([lambda: LStockDailyEnv(df)])

    # policy_kwargs = dict(net_arch=[64, 'lstm', dict(vf=[128, 128, 128], pi=[64, 64])])
    policy_kwargs = dict(net_arch=[128, 'lstm', dict(vf=[256, 256], pi=[256, 256])])
    
    model = A2C('MlpLstmPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=100000)
    model.save('ppo_stock')

    eval_env = DummyVecEnv([lambda: LStockDailyEnv(StockDataFrame(eval_df))])
    # episode_rewards, _  = evaluate_policy(model, eval_env, n_eval_episodes=1, render=True, return_episode_rewards=True) # EVAL_EPS

    # is_recurrent = model.policy.recurrent
    obs = eval_env.reset()

    net_worths = []
    actions = []
    done, state = False, None
    # while not done:
    for _ in range(NEXT_OBSERVATION_SIZE, eval_df.shape[0]):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, _info = eval_env.step(action)
        net_worths.append(_info[0]['net_worth'])
        # if is_recurrent:
        #     obs[0, :] = new_obs
        # else:
        #     obs = new_obs

        # if action[0] < Actions.Buy: # Buy
        #     actions.append(1)
        # elif action[0] < Actions.Sell: # Sell
        #     actions.append(2)
        # else:
        #     actions.append(0)
        actions.append(action[0])
        eval_env.render()

    print(net_worths)
    plt.plot(net_worths)
    plt.show()

    # rewards = []
    # actions = []
    # net_worths = []
    # # for i in range(220):
    # for i in range(NEXT_OBSERVATION_SIZE, df2.shape[0]):
    #     actual_obs = observation(df2, i)
    #     action, _states = model.predict(actual_obs)
    #     action = [action]
    #     obs, reward, done, info = env.step(action)
    #     rewards.append(reward)
    #     actions.append(action[0][0])
    #     net_worths.append(info[0]['net_worth'])
    #     env.render()

    # fig, ax = plt.subplots()
    # ax.plot(rewards, label='rewards')
    # ax.plot(actions, label='actions')
    # ax.legend()
    # ax2 = ax.twinx()
    # ax2.plot(net_worths, label='net worth')
    # ax2.legend()
    # plt.show()

    


if __name__ == '__main__':
    test2()

    # tensorflow 1.14