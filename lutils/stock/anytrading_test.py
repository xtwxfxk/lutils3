# import gym
# import gym_anytrading

# env = gym.make('forex-v0')
# # env = gym.make('stocks-v0')

# from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL

# custom_env = gym.make('forex-v0',
#                df = FOREX_EURUSD_1H_ASK,
#                window_size = 10,
#                frame_bound = (10, 300),
#                unit_side = 'right')

# # custom_env = gym.make('stocks-v0',
# #                df = STOCKS_GOOGL,
# #                window_size = 10,
# #                frame_bound = (10, 300))

# print("env information:")
# print("> shape:", env.shape)
# print("> df.shape:", env.df.shape)
# print("> prices.shape:", env.prices.shape)
# print("> signal_features.shape:", env.signal_features.shape)
# print("> max_possible_profit:", env.max_possible_profit())

# print()
# print("custom_env information:")
# print("> shape:", custom_env.shape)
# print("> df.shape:", env.df.shape)
# print("> prices.shape:", custom_env.prices.shape)
# print("> signal_features.shape:", custom_env.signal_features.shape)
# print("> max_possible_profit:", custom_env.max_possible_profit())


# env.reset()
# env.render()


import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions 
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt

env = gym.make('forex-v0', frame_bound=(50, 100), window_size=10)
# env = gym.make('stocks-v0', frame_bound=(50, 100), window_size=10)

observation = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.render_all()
plt.show()