import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')


x = np.linspace(-1.2,0.6,30)#dis
y = np.linspace(-0.07,0.07,30)#vel


Qs = np.random.randn(x.shape[0],y.shape[0],env.action_space.n)

def choose_action(s,epsilon=0.1):
    p = np.random.rand()
    if p < epsilon:
        return np.random.choice(env.action_space.n)
    else:
        return np.argmax(Qs[s[0],s[1]])

episodes = 10_000
lr = 0.05
gamma = 0.95

epsilon = 0.1

for e in range(episodes):
    s = env.reset()
    sd = np.digitize(s[0],x),np.digitize(s[1],y)
    done = False
    
    if e % 100 == 0:
        print('e',e)
        print(epsilon)
    steps = 0
    
    while not done:
        steps +=1

        a = choose_action(sd,epsilon)
        next_s,r,done,info = env.step(a)
        next_sd = np.digitize(next_s[0],x),np.digitize(next_s[1],y)
        if done:
            target = r
        else:
            target = r + gamma*np.max(Qs[next_sd[0],next_sd[1]])
            
        Qs[sd[0],sd[1],a] += lr*(target - Qs[sd[0],sd[1],a])
        sd = next_sd
        if e%100 == 0:
            env.render()
        if steps >= 10500:
            done = True
