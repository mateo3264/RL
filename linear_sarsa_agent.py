#dyna agent without backward search control heuristic
import numpy as np
from time import sleep,time
import matplotlib.pyplot as plt
import random
rows = 16
cols = 19
start = (5,0)
goal = (0,18)
blocks = [(1,1),(2,1),(3,1),(4,1),(5,1)]#,(0,17),(1,17),(2,17),(3,17),(4,17),(5,17),(9,17),(10,17),(11,17),(12,17)]
doors = [(0,1)]#,(5,7),(6,7),(7,7),(8,7)]
BEHAVIORS = ('U','D','L','R')

#np.random.seed(1)

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

    def give_reward2(self):
        if self.state==self.goal:
            return 100
        elif self.state in doors:
            return 10
        else:
            return -1
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
                return  0
        return 0
    def euclidean_distance_to_door(self,state,behavior):
        next_doors = [door for door in doors if door[1]>state[1]]
        nds = []
        min_dis = float('inf')
        min_dis_x = min_dis_y = 0
        for d in next_doors:
            delta_x = state[1]-d[1]
            delta_y = state[0]-d[0]
            #ed = np.sqrt((delta_x**2)+(delta_y**2))
            if delta_x<min_dis:
                min_dis = delta_x
                min_dis_x,min_dis_y = delta_x,delta_y
#        try:
#            ed = min(nds)
#        except:
#            ed = 1
#        if ed==0:
#            ed = 1
#        den = np.sqrt((rows-1)**2+(cols-1)**2)
        #print('x',min_dis_x,'y',min_dis_y)
        return min_dis_x,min_dis_y

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
        self.weights = np.random.randn(45)/np.sqrt(45)
        self.accumulated_reward = 0
    def __str__(self):
        return 'Linear SARSA Agent'

    def sa2x(self,s,a):
        dist_x,dist_y = self.env.euclidean_distance_to_door(s,a)
        return np.array([
          s[0] - (rows-1)//2              if a == 'U' else 0,
          s[1] - (cols-1)//2            if a == 'U' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'U' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'U' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'U' else 0,
          abs(self.env.state[0]-self.env.goal[0])/(rows-1) if a == 'U' else 0,
          abs(self.env.state[1]-self.env.goal[1])/(cols-1) if a == 'U' else 0,
#          self.env.euclidean_distance_to_door(s,a) if a == 'U' else 0,
          dist_x if a == 'U' else 0,
          dist_y if a == 'U' else 0,
          self.env.is_move_unblocked(s,a) if a == 'U' else 0,
          1                     if a == 'U' else 0,
          s[0] - (rows-1)//2              if a == 'D' else 0,
          s[1] - (cols-1)//2            if a == 'D' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'D' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'D' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'D' else 0,
          abs(self.env.state[0]-self.env.goal[0])/(rows-1) if a == 'D' else 0,
#          self.env.euclidean_distance_to_door(s,a) if a == 'D' else 0,
          abs(self.env.state[1]-self.env.goal[1])/(cols-1) if a == 'D' else 0,
          dist_x if a == 'D' else 0,
          dist_y if a == 'D' else 0,
          self.env.is_move_unblocked(s,a) if a == 'D' else 0,
          1                     if a == 'D' else 0,
          s[0] - (rows-1)//2              if a == 'L' else 0,
          s[1] - (cols-1)//2            if a == 'L' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'L' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'L' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'L' else 0,
          abs(self.env.state[0]-self.env.goal[0])/(rows-1) if a == 'L' else 0,
          abs(self.env.state[1]-self.env.goal[1])/(cols-1) if a == 'L' else 0,
#          self.env.euclidean_distance_to_door(s,a) if a == 'L' else 0,
          dist_x if a == 'L' else 0,
          dist_y if a == 'L' else 0,
          self.env.is_move_unblocked(s,a) if a == 'L' else 0,
          1                     if a == 'L' else 0,

          s[0] - (rows-1)//2    if a == 'R' else 0,
          s[1] - (cols-1)//2            if a == 'R' else 0,
          (s[0]*s[1] - ((rows-1)*(cols-1))//2)/((rows-1)*(cols-1))//2     if a == 'R' else 0,
          (s[0]*s[0] - ((rows-1)*(rows-1))//2)/((rows-1)*(rows-1))//2     if a == 'R' else 0,
          (s[1]*s[1] - ((cols-1)*(cols-1))//2)/((cols-1)*(cols-1))//2     if a == 'R' else 0,
          abs(self.env.state[0]-self.env.goal[0])/(rows-1) if a == 'R' else 0,
          abs(self.env.state[1]-self.env.goal[1])/(cols-1) if a == 'R' else 0,
#          self.env.euclidean_distance_to_door(s,a) if a == 'R' else 0,
          dist_x if a == 'R' else 0,
          dist_y if a == 'R' else 0,
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
lr = 0.1
gamma = .999
a = LinearAgent(env,gamma,lr)

num_episodes = 10000
eps = 0.5
t = 1
t2 = 1
episodes = []
accumulated_rewards = []
descripcion = input('Description: ')
print(descripcion)
print('Agent:',a)
print('rows',rows)
print('# of obstacles',len(blocks))
print('cols',cols)
print('lr',lr)
print('gamma',gamma)
print('eps',eps)
ti = time()
print('tiempo inicial',ti)
promedio = 0
promedios = [promedio]
promedio100 = []
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

    #print('e',it,len(episode))
    promedio100.append(len(episode))
    episodes.append(len(episode))
    promedio += (1/len(promedio100))*(len(episode)-promedio)
    promedios.append(promedio)
    promedio100.append(promedio)
    accumulated_rewards.append(a.accumulated_reward)
    if it%100 == 0 and it !=0:
        t2 +=.01


        a.lr =lr/t2
    if it%100 == 0 and it != 0:

        t +=.1
        print(20*'*')
        print('it',it)
        print('mean',promedio)
        #print('length of episode',len(episode))
        print('accumulated reward',a.accumulated_reward)
        print('eps',eps/t)
        promedio100 = [len(episode)]
        promedio = 0

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
plt.plot([x for x in range(len(promedios))],promedios)
plt.title('promedio episodios')
plt.show()
