import gym
import numpy as np
import random
import math
from time import sleep

env = gym.make('CartPole-v0') # "Cart-Pole" 환경의 초기화

## 환경의 상수 정의
# state dimension 당 discrete states (bucket) 의 수
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# discrete actions 의 수
NUM_ACTIONS = env.action_space.n # (left, right)
# discrete 상태에 따른 bounds설정
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

ACTION_INDEX = len(NUM_BUCKETS)

## Q-Table 생성, 각각의 (상태-액션)을 저장
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning rate
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

## 시뮬레이션을 위한 상수 정의
NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = True

def simulate():
    # learning rate 관련 초기화
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # 환경이 바뀌지 않으므로

    num_streaks = 0

    for episode in range(NUM_EPISODES):

        obv = env.reset() # 환경 리셋
        state_0 = state_to_bucket(obv) # 초기상태

        for t in range(MAX_T):
            env.render()

            action = select_action(state_0, explore_rate) # 행동 선택
            obv, reward, done, _ = env.step(action) # 액션 실행
            state = state_to_bucket(obv) # 결과 관찰

            # q 기반한 결과를 업데이트
            best_q = np.amax(q_table[state])
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            state_0 = state # 다음 상태를 설정

            # 정보 출력
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)

                print("")

            if done:
               print("%f 번의 steps 이후에 Episode %d이 끝났음" % (t, episode))
               if (t >= SOLVED_T):
                   num_streaks += 1
               else:
                   num_streaks = 0
               break

        # 120번을 연속으로 성공하면 넘기면 끝남
        if num_streaks > STREAK_TO_END:
            break

        # 파라메터 업데이트
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


def select_action(state, explore_rate):
    if random.random() < explore_rate:
        action = env.action_space.sample() # 임의의 행동을 선택
    else:
        action = np.argmax(q_table[state]) # 가장 높은 q의 행동을 선택
    return action


def get_explore_rate(t):
    if t >= 24:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
    else:
        return 1.0

def get_learning_rate(t):
    if t >= 24:
         return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
    else:
         return 1.0

def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # state bounds 를 bucket 배열에 매핑
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    simulate()
