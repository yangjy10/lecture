import gym
env = gym.make('CartPole-v0')

print(env.action_space)
print(env.observation_space)

print(env.observation_space.high)
print(env.observation_space.low)

for i_episode in range(20):   # 에피소드 20번
    observation = env.reset() # 리셋으로 초기 관찰값을 정함
    for t in range(100):      # Timestep 게임 진행길이
        env.render()          # 환경을 화면으로 출력
        print(observation)    # 행동 전 환경에서 얻은 관찰값 출력
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # 행동 취득 이후에 얻은 값(관찰값, 보상, 완료여부, 디버깅 정보)
        print("observation:", observation, "Action", action, "Reward:", reward, "Info:", info)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
