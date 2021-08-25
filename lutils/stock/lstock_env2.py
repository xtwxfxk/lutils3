import random
import gym
from gym import error, spaces
import pandas as pd
import numpy as np
from stockstats import StockDataFrame
# from tensorboardX import SummaryWriter

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_NUM_AMOUNTS = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 60
MAX_STEPS = 240 # 40000
NEXT_OBSERVATION_SIZE = 6

INITIAL_ACCOUNT_BALANCE = 10000

# writer = SummaryWriter('log')

class LStockDailyEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df=None):
        super(LStockDailyEnv, self).__init__()

        # self.days = []
        # self.step_index = 1
        self.df = df
        # row_index = 0
        # while row_index < df.shape[0]:
        #     self.days.append(df[row_index:row_index+240])
        #     row_index = row_index + 240
        # if self.days[-1].shape[0] < 240:
        #     self.days.pop(-1)

        # self.current_step = 0
        # self.current_day = 1
        # # self.day = random.choice(self.days)
        # self.yesterday = self.days[self.current_day - 1]
        # self.day = pd.concat([self.yesterday[:], self.days[self.current_day]])

        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.reward_range = (0, 1)
        # self.reward_range = (0, 100)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(5, NEXT_OBSERVATION_SIZE), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['open'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['high'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['low'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['close'].values / MAX_SHARE_PRICE,
            # self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['volume'].values / MAX_NUM_SHARES,
            # self.df['macd'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['macdh'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            # self.df['macds'][self.current_step: self.current_step + NEXT_OBSERVATION_SIZE].values,
            self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['open'].values,
            self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['high'].values,
            self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['low'].values,
            self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['close'].values,
            self.df.iloc[self.current_step: self.current_step + NEXT_OBSERVATION_SIZE]['volume'].values,
        ])

        return frame

        # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [[
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.shares_held / MAX_NUM_SHARES,
        #     self.cost_basis / MAX_SHARE_PRICE,
        #     self.total_shares_sold / MAX_NUM_SHARES,
        #     self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        # ]], axis=0)

        # return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.iloc[self.current_step]['close'] # + 0.02
        action_type = action[0]
        amount = action[1]
        # amount = 1

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

        # writer.add_scalar('Net Worth', self.net_worth, self.step_index)

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0


    def step(self, action):
        # Execute one time step within the environment
        # self.step_index = self.step_index + 1
        shares_held = self.shares_held
        self._take_action(action)

        self.current_step = self.current_step + 1

        if self.current_step > self.df.shape[0] - NEXT_OBSERVATION_SIZE:
            self.current_step = 0

        # delay_modifier = (self.current_step / MAX_STEPS)
        # reward = self.balance - INITIAL_ACCOUNT_BALANCE
        # reward = self.balance * delay_modifier
        # done = self.net_worth <= 0
        # done = self.current_step >= (self.day.shape[0]-MAX_OPEN_POSITIONS) or self.net_worth <= 0
        # reward = 1 if self.net_worth > self.balance else 0
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        # if reward < 0: reward = 0
        # if done:
        #     # reward = 1 if self.net_worth > self.balance else 0
        #     self.balance = self.net_worth
        # else:
        #     reward = 0

        done = self.net_worth <= INITIAL_ACCOUNT_BALANCE * .7

        obs = self._next_observation()

        reward = 0

        action_type = action[0]
        if action_type < 1: # Buy
            reward = 1 if obs[:-1, 3].mean() >= self.df.iloc[self.current_step]['close'] else 0
        elif action_type >= 1 and action_type < 2: # Sell
            reward = 1 if obs[:-1, 3].mean() <= self.df.iloc[self.current_step]['close'] and shares_held > self.shares_held else 0
        else:
            reward = 1 if obs[:-1, 3].mean() >= self.df.iloc[self.current_step]['close'] else 0
        # reward = 1 if obs[:-1, 3].mean() - self.df.iloc[self.current_step]['close'] > 0 else 0

        return obs, reward, done, {'net_worth': self.net_worth, 'current_step': self.current_step}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # self.day = random.choice(self.days)
        # self.current_step = MAX_OPEN_POSITIONS

        # self.current_day = self.current_day + 1
        # if self.current_day >= len(self.days):
        #     self.current_day = 1

        # self.yesterday = self.days[self.current_day - 1]
        # self.day = pd.concat([self.yesterday[-MAX_OPEN_POSITIONS:], self.days[self.current_day]])

        self.current_step = random.randint(0, self.df.shape[0] - NEXT_OBSERVATION_SIZE)

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




# def observation(df, current_step):
#     frame = np.array([
#         df.iloc[current_step - NEXT_OBSERVATION_SIZE: current_step]['open'].values / MAX_SHARE_PRICE,
#         df.iloc[current_step - NEXT_OBSERVATION_SIZE: current_step]['high'].values / MAX_SHARE_PRICE,
#         df.iloc[current_step - NEXT_OBSERVATION_SIZE: current_step]['low'].values / MAX_SHARE_PRICE,
#         df.iloc[current_step - NEXT_OBSERVATION_SIZE: current_step]['close'].values / MAX_SHARE_PRICE,
#         df.iloc[current_step - NEXT_OBSERVATION_SIZE: current_step]['vol'].values / MAX_NUM_SHARES,
#     ])
#     return frame
#     # Append additional data and scale each value to between 0-1
#     obs = np.append(frame, [[
#         balance / MAX_ACCOUNT_BALANCE,
#         max_net_worth / MAX_ACCOUNT_BALANCE,
#         shares_held / MAX_NUM_SHARES,
#         cost_basis / MAX_SHARE_PRICE,
#         total_shares_sold / MAX_NUM_SHARES,
#         total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
#     ]], axis=0)

#     return obs


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
    from sklearn import preprocessing

    import pandas as pd

    from lutils.stock import LTdxHq

    ltdxhq = LTdxHq()
    df = ltdxhq.get_k_data_1min('000032') # 000032 300142 603636 
    # df = StockDataFrame(df.rename(columns={'vol': 'volume'}))
    min_max_scaler = preprocessing.MinMaxScaler()
    df = pd.DataFrame(min_max_scaler.fit_transform(df.drop(columns=['date', 'code'])))
    df.columns = ['open', 'close', 'high', 'low', 'volume', 'amount']

    ltdxhq.close()
    # df = ltdxhq.get_k_data_5min('603636')
    # df = ltdxhq.get_k_data_daily('603636')

    df1 = df[:-240]
    df2 = df[-240:]
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: LStockDailyEnv(df1)])
    # model = PPO2(MlpPolicy, env, verbose=1) # , tensorboard_log='log')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='log')
    model.learn(100000)
    # model = PPO1(LstmPolicy, env, verbose=1)
    # model.learn(total_timesteps=1000)

    env.set_attr('df', df2)
    obs = env.reset()

    rewards = []
    actions = []
    net_worths = []
    # for i in range(220):
    for i in range(NEXT_OBSERVATION_SIZE, df2.shape[0]):
        # actual_obs = observation(df2, i)
        # action, _states = model.predict(actual_obs)
        # action = [action]
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action[0][0])
        net_worths.append(info[0]['net_worth'])
        # print(info[0]['current_step'])
        env.render()

    fig, ax = plt.subplots()
    ax.plot(rewards, label='rewards')
    ax.plot(actions, label='actions')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(net_worths, label='net worth', color='red')
    ax2.legend()
    plt.show()

    # tensorflow 2.6

def random_train_model():

    import gym
    import datetime as dt
    import matplotlib.pyplot as plt

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    import pandas as pd

    from lutils.stock import LTdxHq

    import tushare as ts
    pro = ts.pro_api()

    stock_codes = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')

    env = DummyVecEnv([lambda: LStockDailyEnv()])
    # model = PPO('MlpPolicy', env, verbose=1)
    model = PPO.load('ppo_stock')
    model.set_env(env)
    for i in range(10):
        code = random.choice(stock_codes['ts_code'])[:-3]
        print('load data: %s' % code)
        ltdxhq = LTdxHq()
        df = ltdxhq.get_k_data_1min(code) # 000032 300142 603636 
        ltdxhq.close()

        df = df[:-240]

        env.set_attr('df', df)
        env.reset()
        model.learn(20000)

    model.save('ppo_stock')

def random_train_test():
    import gym
    import datetime as dt
    import matplotlib.pyplot as plt

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    import pandas as pd

    from lutils.stock import LTdxHq

    ltdxhq = LTdxHq()
    df = ltdxhq.get_k_data_1min('000032') # 000032 300142 603636 
    df = df[-240:]
    ltdxhq.close()

    model = PPO.load('ppo_stock')

    env = DummyVecEnv([lambda: LStockDailyEnv(df)])
    obs = env.reset()

    rewards = []
    actions = []
    net_worths = []
    for i in range(NEXT_OBSERVATION_SIZE, df.shape[0]):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action[0][0])
        net_worths.append(info[0]['net_worth'])
        # print(info[0]['current_step'])
        env.render()

    fig, ax = plt.subplots()
    ax.plot(rewards, label='rewards')
    ax.plot(actions, label='actions')
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(net_worths, label='net worth', color='red')
    ax2.legend()
    plt.show()


if __name__ == '__main__':
    test_rl()
    # random_train_model()
    # random_train_test()
