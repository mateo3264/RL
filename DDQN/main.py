#The code is mostly based on phil https://www.youtube.com/watch?v=UCgsv6tMReY&t=2207s

import envs
import numpy as np
from DDQN import DDQNAgent


if __name__ == '__main__':
    
    
    #env = envs.Env(30,40,[(1,1)],{(0,3):1,(1,3):-1})
    env = envs.ShmupEnv(12,12,5)
    ddqn_agent = DDQNAgent(alpha=0.0001,gamma=0.99,
                           n_actions=4,epsilon=1.0,epsilon_dec=1,
                           batch_size=64,input_dims=12)
    n_games = 1500
    print('---TRAINING STARTED---')
    ddqn_scores = []
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        ddqn_agent.epsilon *=0.9
        if i%150 == 0:
            
            ddqn_agent.alpha = alphas[i]
        while not done:
        
            action = ddqn_agent.choose_action(observation)
            observation_,reward,done = env.step(action)
            observation_ = np.array(observation_)
            score += reward
            ddqn_agent.remember(observation,action,reward,observation_,done)
            observation = observation_
            ddqn_agent.learn()
            if i%10 == 0 and i>0:
                env.render()
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0,i-100):(i+1)])
        print('episode',i,'epsilon ',ddqn_agent.epsilon,'alpha ',ddqn_agent.alpha,'score %.2f'%score,'average score %.2f'%avg_score)

        if i%10 == 0 and i > 0:
            ddqn_agent.save_model()
    
    filename = 'shmup.png'

import matplotlib.pyplot as plt
plt.plot(ddqn_scores)
plt.show()
                                
