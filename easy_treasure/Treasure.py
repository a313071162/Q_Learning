# -*- encoding: utf-8 -*-


"""
@File    : Treasure.py
@Time    : 2020/8/4 上午9:15
@Author  : dididididi
@Email   : 
@Software: PyCharm
"""

import numpy as np
import pandas as pd
import time


np.random.seed(2)

N_STATE = 6  # the length of treasure
ACTIONS = ['left', 'right']  # action
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
LAMBDA = 0.9  # discount factor
MAX_EPISODES = 13  # max episodes
FRESH_TIME = 0.3  # fresh time for one move



# 1、初始化q表
def build_q_table(n_states, actions):
    """
    初始化q表
    :param n_states: 运行步数
    :param actions: 采取动作2
    :return:
    """
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    print(table)
    return table



# 2、选动作
def choose_action(state, q_table):
    """
    选择动作
    :return:
    """
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = ACTIONS[state_actions.argmax()]
    return action_name


# 3、获取agent与环境之间的反馈
def get_env_return(S, A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


# 4、更新环境
def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATE - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_step = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


if __name__ == '__main__':
    q_table = build_q_table(N_STATE, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminaled = False
        # 更新环境
        update_env(S, episode, step_counter)
        while not is_terminaled:
            # 选取行为
            A = choose_action(S, q_table)
            # 获取环境回报
            S_, R = get_env_return(S, A)
            q_predict = q_table.iloc[S, q_table.columns.get_loc(A)]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminaled = True
            # 更新S
            q_table.iloc[S, q_table.columns.get_loc(A)] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter)
            step_counter = step_counter + 1

    print(q_table)
