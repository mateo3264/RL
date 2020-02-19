#dyna agent without backward search control heuristic
import numpy as np
from time import sleep,time
import matplotlib.pyplot as plt
import random
rows = 6
cols = 9
start = (2,0)
goal = (0,8)
blocks = [(1,1),(2,1),(3,1),(4,1),(5,1),(0,7),(1,7),(2,7),(3,7),(4,7)]#,(5,7),(9,7),(10,7),(11,7),(12,7)]

BEHAVIORS = ('U','D','L','R')


class Environment:
    def __init__(self):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal
        self.blocks = blocks
        self.state = self.start
        self.iterations = 0
        self.unblocked_move = -1
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
                #self.unblocked_move = 1
        return self.state

    def give_reward(self):
        return 1 if self.state==self.goal else -0.01
    def is_move_unblocked(self,state,behavior):
        r = state[0]
        c = state[1]
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
                #self.state = (r,c)
                return  1
            else:
                return  -1
        return -1

    def reset(self):
        self.state = self.start

def not_in_blocks(state):
    bs = [list(block) for block in blocks]
    return list(state) in bs
def in_bounds(state):
    return True if 0<=state<=ROWS-1 and 0<=state<=COLS-1 else False
def neighbors(state):
    ns = [state+np.array([1,0]),state-np.array([1,0]),state+np.array([0,1]),state-np.array([0,1])]
    ns = list(filter(not_in_blocks,state))
    ns = list(filter(in_bounds,state))
    return ns

def neighbors2(state,action):
    neighbor = 0

class LinearAgent:
    def __init__(self,env,gamma,learning_rate):
        self.env = env
        self.gamma = gamma
        self.lr = learning_rate
        self.weights = np.random.randn(37)/np.sqrt(37)
        self.accumulated_reward = 0
    def __str__(self):
        return 'Linear SARSA Agent'

    def sa2x(self,s,a):
        return np.array([
          s[0] - (rows-1)//2              if a == 'U' else 0,
          s[1] - (cols-1)//2            if a == 'U' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'U' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'U' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'U' else 0,
          abs(1/(self.env.state[0]-self.env.goal[0]))/(rows-1) if a == 'U' else 0,
          abs(1/(self.env.state[1]-self.env.goal[1]))/(cols-1) if a == 'U' else 0,
          self.env.is_move_unblocked(s,a) if a == 'U' else 0,
          1                     if a == 'U' else 0,
          s[0] - (rows-1)//2              if a == 'D' else 0,
          s[1] - (cols-1)//2            if a == 'D' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'D' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'D' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'D' else 0,
          abs(1/(self.env.state[0]-self.env.goal[0]))/(rows-1) if a == 'D' else 0,
          abs(1/(self.env.state[1]-self.env.goal[1]))/(cols-1) if a == 'D' else 0,
          self.env.is_move_unblocked(s,a) if a == 'D' else 0,
          1                     if a == 'D' else 0,
          s[0] - (rows-1)//2              if a == 'L' else 0,
          s[1] - (cols-1)//2            if a == 'L' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'L' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'L' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'L' else 0,
          abs(1/(self.env.state[0]-self.env.goal[0]))/(rows-1) if a == 'L' else 0,
          abs(1/(self.env.state[1]-self.env.goal[1]))/(cols-1) if a == 'L' else 0,
          self.env.is_move_unblocked(s,a) if a == 'L' else 0,
          1                     if a == 'L' else 0,

          s[0] - (rows-1)//2    if a == 'R' else 0,
          s[1] - (cols-1)//2            if a == 'R' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'R' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'R' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'R' else 0,
          abs(1/(self.env.state[0]-self.env.goal[0]))/(rows-1) if a == 'R' else 0,
          abs(1/(self.env.state[0]-self.env.goal[0]))/(cols-1) if a == 'R' else 0,
          self.env.is_move_unblocked(s,a) if a == 'R' else 0,
          1                     if a == 'R' else 0,
          1
        ])
    def get_best_action(self,s):
        best_behavior = None
        best_value = float('-inf')
        for b in BEHAVIORS:
            q_sa = self.predict(s,b)
            if q_sa>best_value:
                best_value = q_sa
                best_behavior = b
        return best_behavior,best_value
    def explore(self,behavior,eps=0.5):
        p = np.random.random()
        if p<eps:
            return np.random.choice(BEHAVIORS)
        else:
            return behavior
    def predict(self,s,a):
        x = self.sa2x(s,a)
        prediction = np.dot(self.weights,x)
        #print('w',self.weights)
        #print('pred',prediction)
        return prediction
    def grad(self,s,a):
        x = self.sa2x(s,a)
        return x
    def learn(self,s,a,eps):
        #self.lr = self.lr/1.1
        #a = self.get_best_action(s)
        #a = self.explore(a,eps)
        next_state = self.env.next_state(a)
        #next_a,next_value = self.get_best_action(next_state)

        reward = self.env.give_reward()
        self.accumulated_reward +=reward
        next_best_action,next_best_value = self.get_best_action(next_state)
        next_a = self.explore(next_best_action)
        if next_state != self.env.goal:
            target = reward + self.gamma*self.predict(next_state,next_a) - self.predict(s,a)
        else:
            target = reward - self.predict(s,a)
        self.weights += self.lr*target*self.grad(s,a)
        #next_a = self.explore(next_best_action)
        return next_state,next_a
    def reset(self):
        self.accumulated_reward = 0

env = Environment()
lr = 0.01
gamma = 0.9
a = LinearAgent(env,gamma,lr)

num_episodes = 10000
eps = 0.3
t = 1
episodes = []
accumulated_rewards = []
print('Agent:',a)
print('rows',rows)
print('# of obstacles',len(blocks))
print('cols',cols)
print('lr',lr)
print('gamma',gamma)
print('eps',eps)
ti = time()
print('tiempo inicial',ti)
for it in range(num_episodes):
    s = env.start
    env.reset()
    a.reset()
    episode = [s]
    #print('it',it)
    b,v = a.get_best_action(s)
    b = a.explore(b,eps/t)
    while env.state != env.goal:
        #print('a its',a.iterations)

        s,b = a.learn(s,b,eps)
        #s = env.state
        #print('s',s)
        episode.append(s)

    print('e',it,len(episode))
    episodes.append(len(episode))
    accumulated_rewards.append(a.accumulated_reward)
    if it%100 == 0 and it != 0:
        t +=.1
        print('length of episode',len(episode))
        print('accumulated reward',a.accumulated_reward)
        print('eps',eps/t)

        a.lr =lr/t
        #sleep(1)
        try:
            print(episodes[-1])
        except:
            pass
print('tiempo total',time()-ti)
plt.plot([x for x in range(num_episodes)],episodes)
plt.title('longitud de episodios')
plt.show()
plt.plot([x for x in range(num_episodes)],accumulated_rewards)
plt.title('refuerzo acumulado')
plt.show()
