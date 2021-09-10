import random
import gym
from gym import error, spaces
import pandas as pd
import numpy as np
from enum import Enum
from stockstats import StockDataFrame
# from tensorboardX import SummaryWriter

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_NUM_AMOUNTS = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 60
MAX_STEPS = 240 # 40000
NEXT_OBSERVATION_SIZE = 10

INITIAL_ACCOUNT_BALANCE = 10000

# writer = SummaryWriter('log')

class Actions(Enum):
    Hold = 0
    Sell = 1
    Buy = 2

class LStockDailyEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None):
        super(LStockDailyEnv, self).__init__()

        self.df = df

        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # self.reward_range = (0, 1)
        # self.reward_range = (0, 100)
        self.reward_range = (-np.inf, np.inf)

        self.action_space = spaces.Discrete(len(Actions))

        self.observation_space = spaces.Box(low=0, high=1, shape=(13, NEXT_OBSERVATION_SIZE), dtype=np.float32)

    def seed(self, seed=None):
        pass


    def _next_observation(self):
        if self.df is None:
            return None

        # for i in range(NEXT_OBSERVATION_SIZE)
        #     self.df[:-self.current_step +].describe()

        frame = np.array([ # 11 * 10
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['open'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['high'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['low'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['close'].values / MAX_SHARE_PRICE,
            self.df.iloc[self.current_step - NEXT_OBSERVATION_SIZE: self.current_step]['volume'].values / MAX_NUM_SHARES,
            # self.df['close'].pct_change().fillna(0)[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE],

            self.df['macd'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            self.df['macdh'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            self.df['macds'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            # # self.df['volume_delta'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['open_2_d'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['open_-2_r'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['cr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['cr-ma1'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['cr-ma2'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['cr-ma3'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            self.df['kdjk'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            self.df['kdjd'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            self.df['kdjj'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].values,
            self.df['rsi_6'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            self.df['rsi_12'][self.current_step - NEXT_OBSERVATION_SIZE: self.current_step].fillna(0).values,
            # self.df['open_2_sma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['dma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['pdi'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['mdi'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['dx'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['adx'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # self.df['adxr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['tema'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # # self.df['vr'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].fillna(0).values,
            # # self.df['vr_6_sma'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,

            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['open'].values,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['high'].values,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['low'].values,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['close'].values,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['volume'].values,
        ])

        return frame

    def _take_action(self, action):
        if self.df is not None:
            current_price = self.df.iloc[self.current_step]['close'] # + 0.02

        # action_type = action[0]
        # amount = action[1]
        # amount = 1

        if action == Actions.Buy.value:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action == Actions.Sell.value:
            shares_sold = int(self.shares_held)
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
        shares_held = self.shares_held
        self._take_action(action)

        self.current_step = self.current_step + 1

        done = self.net_worth <= INITIAL_ACCOUNT_BALANCE * .9 or self.current_step > self.step_range.stop

        obs = self._next_observation()

        reward = 0
        if self.df is not None:
            if action == Actions.Buy.value: # Buy
                reward = self.df.iloc[self.current_step + 1]['close'] - self.df.iloc[self.current_step]['close']
            elif action == Actions.Sell.value: # Sell
                reward = self.df.iloc[self.current_step]['close'] - self.df.iloc[self.current_step + 1]['close']
            else: # Hold
                reward = self.df.iloc[self.current_step + 1]['close'] - self.df.iloc[self.current_step]['close']

        if done:
            reward = self.net_worth - INITIAL_ACCOUNT_BALANCE


        return obs, reward, done, {'net_worth': self.net_worth, 'current_step': self.current_step}

    def reset(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        date_index = np.random.choice(self.df.index.unique(level=0)[1:-1])
        self.day = self.df.loc[date_index]

        self.step_range = self.df.index.get_loc(date_index)
        self.current_step = self.step_range.start

        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')



def test_rl():
    import gym
    import datetime as dt
    import matplotlib.pyplot as plt

    # from stable_baselines.common.policies import MlpPolicy, CnnPolicy, MlpLstmPolicy, ActorCriticPolicy, LstmPolicy
    # from stable_baselines.common.vec_env import DummyVecEnv
    # from stable_baselines import PPO2, PPO1, A2C, DQN, TD3, SAC

    # from stable_baselines3.common.policies import MlpPolicy
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    from sklearn import preprocessing

    import pandas as pd

    from lutils.stock import LTdxHq

    ltdxhq = LTdxHq()
    code = '600519' # 000032 300142 603636 600519
    df = ltdxhq.get_k_data_1min(code, end='2021-09-02') # 000032 300142 603636 600519
    # df = ltdxhq.get_k_data_daily('603636', end='2019-01-01') # 000032 300142 603636 600519
    df = StockDataFrame(df.rename(columns={'vol': 'volume'}))

    # min_max_scaler = preprocessing.MinMaxScaler()
    # df = pd.DataFrame(min_max_scaler.fit_transform(df.drop(columns=['date', 'code'])))
    # df.columns = ['open', 'close', 'high', 'low', 'volume', 'amount']

    df_eval = ltdxhq.get_k_data_1min(code, start='2021-09-01')
    df_eval = StockDataFrame(df_eval.rename(columns={'vol': 'volume'}))

    ltdxhq.close()
    # df = ltdxhq.get_k_data_5min('603636')
    # df = ltdxhq.get_k_data_daily('603636')

    # df1 = df[:-240]
    # df2 = df[-240:]
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: LStockDailyEnv(df)])
    # model = PPO2(MlpPolicy, env, verbose=1) # , tensorboard_log='log')
    model = PPO('MlpPolicy', env, verbose=1) # , tensorboard_log='log')
    model.learn(100000)
    # model = PPO1(LstmPolicy, env, verbose=1)
    # model.learn(total_timesteps=1000)



    # env.set_attr('df', df2)
    # obs = env.reset()

    # rewards = []
    # actions = []
    # net_worths = []
    # # for i in range(220):
    # for i in range(NEXT_OBSERVATION_SIZE, df2.shape[0]):
    #     # actual_obs = observation(df2, i)
    #     # action, _states = model.predict(actual_obs)
    #     # action = [action]
    #     action, _states = model.predict(obs)
    #     obs, reward, done, info = env.step(action)
    #     rewards.append(reward)
    #     actions.append(action[0][0])
    #     net_worths.append(info[0]['net_worth'])
    #     # print(info[0]['current_step'])
    #     env.render()

    # mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=1, render=True) # EVAL_EPS

    # print(mean_reward)

    model.save('ppo_stock')
    # model = PPO.load('ppo_stock')

    eval_env = DummyVecEnv([lambda: LStockDailyEnv(df_eval)])
    obs = eval_env.reset()

    net_worths = []
    actions = []
    done, state = False, None
    while not done:
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

    plt.plot(net_worths)
    plt.plot(actions)
    plt.show()

    # fig, ax = plt.subplots()
    # # ax.plot(rewards, label='rewards')
    # ax.plot(actions, '.', label='actions')
    # # ax.legend()
    # ax2 = ax.twinx()
    # ax2.plot(net_worths, label='net worth', color='red')
    # ax2.legend()
    # plt.show()

    # tensorflow 2.6


if __name__ == '__main__':
    test_rl()
    # random_train_model()
    # random_train_test()
