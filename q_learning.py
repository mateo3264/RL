import numpy as np
from time import sleep
import matplotlib.pyplot as plt

rows = 16
cols = 19
start = (2,0)
goal = (0,18)
blocks = [(1,1),(2,1),(3,1),(4,1),(5,1)]

BEHAVIORS = ('U','D','L','R')


class Environment:
    def __init__(self):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.blocks = blocks
        self.state = self.start
    def next_state(self,behavior):
        r = self.state[0]
        c = self.state[1]
        if behavior == 'U':
            r -=1
        elif behavior == 'D':
            r +=1
        elif behavior == 'L':
            c -=1
        elif behavior == 'R':
            c +=1

        if (0 <= r <= self.rows-1 and 0 <= c <= self.cols-1):
            if (r,c) not in blocks:
                self.state = (r,c)
        return self.state

    def give_reward(self):
        return 1 if self.state==self.goal else -0.01

    def reset(self):
        self.state = self.start


class QAgent:
    def __init__(self,env,learning_rate,gamma):
        self.lr = learning_rate
        self.gamma = gamma
        self.Q = {}
        self.env = env
        self.agent_state = self.env.state
        self.accumulated_reward = 0
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                self.Q[(r,c)] = {}
                for b in BEHAVIORS:
                    self.Q[(r,c)][b] = 0


    def get_best_action(self,s):
        best_behavior = None
        best_value = float('-inf')
        for b in BEHAVIORS:
            if self.Q[s][b]>best_value:
                best_value = self.Q[s][b]
                best_behavior = b
        return best_behavior,best_value
    def explore(self,behavior,eps=0.5):
        p = np.random.random()
        if p<eps:
            return np.random.choice(BEHAVIORS)
        else:
            return behavior
    def learn(self,s,b,eps):
        nextState = self.env.next_state(b)
        self.agent_state = nextState
        reward = self.env.give_reward()
        self.accumulated_reward += reward
        if self.env.state != self.env.goal:
            nextBehavior,nextQValue = self.get_best_action(nextState)
            nextBehavior = self.explore(nextBehavior,eps)
            self.Q[s][b] = self.Q[s][b] + self.lr*(reward +self.gamma*nextQValue - self.Q[s][b])
        else:
            self.Q[s][b] = self.Q[s][b] + self.lr*(reward - self.Q[s][b])

    def reset(self):
        self.accumulated_reward = 0

        #if s == self.env.goal:
            #print('s',s,'goal!!!!')
        #print('b',b)
        #sleep(.1)
        #print('s',s)
        #print('tupla',(reward +self.gamma*nextQValue - self.Q[s][b]))


env = Environment()
a = QAgent(env,1,0.9)

num_episodes = 1000
eps = 0.5
t = 1
episodes = []
accumulated_rewards = []
for it in range(num_episodes):
    s = env.start
    env.reset()
    a.reset()
    episode = [s]
    #print('it',it)

    while env.state != env.goal:
        b,v = a.get_best_action(s)
        b = a.explore(b,eps/t)
        a.learn(s,b,eps)
        s = env.state
        episode.append(s)
    episodes.append(len(episode))
    accumulated_rewards.append(a.accumulated_reward)
    if it%100 == 0 and it != 0:
        t +=.1
        print('length of episode',len(episode))
        print('accumulated reward',a.accumulated_reward)
        print('eps',eps/t)
        #sleep(1)
        try:
            print(episodes[-1])
        except:
            pass
plt.plot([x for x in range(num_episodes)],episodes)
plt.title('longitud de episodios')
plt.show()
plt.plot([x for x in range(num_episodes)],accumulated_rewards)
plt.title('refuerzo acumulado')
plt.show()
