import gym
import numpy as np
import matplotlib.pyplot as plt 
from gym.envs.registration import register
import random as pr

env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n]) # Q-table 초기화

learning_rate = 0.85
dis = .99 # discount 요소
    
# learning 횟수
num_episodes = 2000

# 학습 시 reward 저장
rList = []

for i in range(num_episodes):
    # env 리셋
    state = env.reset()
    rAll = 0
    done = False
    
    e = 1./((i//100)+1) # decaying E-greedy
    
    # Q-테이블 알고리즘
    while not done:
        # egreedy에 의한 행동 설정
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # new_state, reward 업데이트 
        new_state, reward, done, _ = env.step(action)
        
        # 업데이트 Q-table (상태, 행동)
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate*(reward + dis*np.max(Q[new_state, :]))
        
        rAll += reward
        state = new_state

    rList.append(rAll)

print('성공율: ', str(sum(rList)/num_episodes))
print('Q-table')
print(Q)
plt.bar(range(len(rList)), rList, color = 'blue')
plt.show()
