import numpy as np
from time import sleep
import matplotlib.pyplot as plt

rows = 6
cols = 9
start = (2,0)
goal = (0,8)
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
        return 1 if self.state==self.goal else -.01

    def reset(self):
        self.state = self.start
    def choose_initial_state_behavior(self):#exploring starts
        s_r = np.random.choice(self.rows) #state_row
        s_c = np.random.choice(self.cols) #state_col
        while (s_r,s_c) in self.blocks:
            s_r = np.random.choice(self.rows)
            s_c = np.random.choice(self.cols)
        b = np.random.choice(BEHAVIORS)
        self.state = (s_r,s_c)
        self.start = self.state
        return (s_r,s_c),b


class MCAgent:
    def __init__(self,env,learning_rate,gamma):
        self.lr = learning_rate
        self.gamma = gamma
        self.Q = {}
        #self.returns = {}

        self.env = env
        self.agent_state = self.env.state
        self.accumulated_reward = 0
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                self.Q[(r,c)] = {}
                #self.returns[(r,c)] = {}
                for b in BEHAVIORS:
                    self.Q[(r,c)][b] = 0#1:avg G,number
                    #self.returns[(r,c)][b] =
    def __str__(self):
        return 'Monte Carlo Agent'
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
    def learn(self,episode,eps):
        G = 0
        first = True
        for i,(s,b,r) in enumerate(reversed(episode)):
            #if i==1:
                #print('last b-state value',s,self.Q[s][b])
                #first = False
            v_s = self.Q[s][b]
            self.Q[s][b] += self.lr*(G - v_s)
            G = r + self.gamma*G
            #self.
#        nextState = self.env.next_state(b)
#        self.agent_state = nextState
#        reward = self.env.give_reward()
#        self.accumulated_reward += reward
#        if self.env.state != self.env.goal:
#            nextBehavior,nextQValue = self.get_best_action(nextState)
#            nextBehavior = self.explore(nextBehavior,eps)
#            self.Q[s][b] = self.Q[s][b] + self.lr*(reward +self.gamma*nextQValue - self.Q[s][b])
#        else:
#            self.Q[s][b] = self.Q[s][b] + self.lr*(reward - self.Q[s][b])

    def reset(self):
        self.accumulated_reward = 0
        #self.agent_state
        #if s == self.env.goal:
            #print('s',s,'goal!!!!')
        #print('b',b)
        #sleep(.1)
        #print('s',s)
        #print('tupla',(reward +self.gamma*nextQValue - self.Q[s][b]))


env = Environment()
lr = 0.01
gamma = 0.9
a = MCAgent(env,lr,gamma)
print('lr:',lr)
num_episodes = 1000
eps = 0.2
t = 1
episodes = []
accumulated_rewards = []

def test(env,agent,initial_state=(2,3),eps=0.01):
    s = initial_state
    env.state = s
    env.start = s
    a.reset()
    a.agent_state = s
    #print('s',first_state,'b',first_behavior)
    r = env.give_reward()
    #s_b_r_list = [(first_state,first_behavior,0)]
    #it_ep = 0
    episode = []
    last_state = None
    current_state = s
    while env.state != env.goal:
        #print('it of ep',it_ep)
        #it_ep +=1
        b,v = a.get_best_action(s)
        b = a.explore(b,eps)
        #a.learn(s,b,eps)
#        print('s',s)
        env.next_state(b)
#        last_state = current_state
#        current_state = env.state
#        if last_state == current_state:
#            break
        r = env.give_reward()
        a.accumulated_reward += r
        #s_prime = env.state

        #s_b_r_list.append((s_prime,b,r))
        s = env.state
        a.agent_state = s
        episode.append(s)

    return episode


for it in range(num_episodes):

    env.reset()
    a.reset()

    #print('it',it)
    first_state,first_behavior = env.choose_initial_state_behavior()
    s = env.start
    episode = [first_state]
    a.agent_state = first_state
    #print('s',first_state,'b',first_behavior)
    r = env.give_reward()
    s_b_r_list = [(first_state,first_behavior,0)]
    it_ep = 0
    while env.state != env.goal:
        #print('it of ep',it_ep)
        #it_ep +=1
        b,v = a.get_best_action(s)
        b = a.explore(b,eps/t)
        #a.learn(s,b,eps)
        #print('s',s,'b',b)
        env.next_state(b)
        r = env.give_reward()
        s_b_r_list.append((s,b,r))
        a.accumulated_reward += r
        s_prime = env.state


        s = s_prime
        a.agent_state = s
        episode.append(s)
    #print('(0,7)',a.Q[(0,7)])
    #print('(1,7)',a.Q[(1,7)])
    #print('(1,8)',a.Q[(1,8)])
    #print('it',it)
    episode = test(env,a)
    #if len(episode)<=30:

    a.learn(s_b_r_list,eps/t)
    episodes.append(len(episode))
    accumulated_rewards.append(a.accumulated_reward)
    if it%50 == 0 and it != 0:
        t +=.1
        #print('length of episode',len(episode))
        print(it,'len test episode --->',len(episode))
        if len(episode)<=30:
            print(episode)
        #np.mean(episodes[it:])
        print('accumulated reward',a.accumulated_reward)
        print('eps',eps/t)
        #sleep(1)
        try:
            print(episodes[-1])
        except:
            pass
plt.plot([x for x in range(num_episodes)],episodes)
plt.title(str(a)+' : '+'longitud de episodios')
plt.show()
plt.plot([x for x in range(num_episodes)],accumulated_rewards)
plt.title(str(a)+' : '+'refuerzo acumulado')
plt.show()
print('fin')
test(env,a)
