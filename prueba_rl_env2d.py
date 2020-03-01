import numpy as np
from time import sleep
from ttkthemes import ThemedTk as tk
from tkinter import ttk

class Window:
    def __init__(self):
        self.root = tk(theme='radiance')
        #self.root.get_themes()
        #self.root.set_theme('radiance')
        self.b = ttk.Button(self.root,text='hola')
        self.b.pack()
        self.root.mainloop()
#w = Window()

Y_REGION = 6
X_REGION = 10
INITIAL_STATE = (0.5,0.5)
GOAL = ((4.3,5.3),(8.9,9.9))
OBSTACLES = []

class Environment:
    def __init__(self,y_region,x_region,initial_state,goal):
        self.y_region = y_region
        self.x_region = x_region
        self.initial_state = initial_state
        self.state = initial_state
        self.goal = goal
        self.goal_center =(self.goal[0][0] + self.goal[0][1])//2,(self.goal[1][0] + self.goal[1][1])//2
        self.terminal = False
        #self.magnitud_of_movement =
    def next_state(self,behavior):
        y = 0
        x = 0
        if behavior == 0:
            y =-1
        elif behavior == 1:
            x =-1
        elif behavior == 2:
            y = 1
        elif behavior == 3:
            x = 1
        if 0 <= self.state[1] + x <= X_REGION and 0 <= self.state[0] + y <= Y_REGION:
            #sección para colocar código que verifique si el agente esta dentro de
            #una región de un bloque
            self.state = (self.state[0]+y ,self.state[1]+x)
        return self.state

    def get_reward(self):
        if self.goal[0][0]<=self.state[0]<=self.goal[0][1] and \
            self.goal[1][0]<=self.state[1]<=self.goal[1][1]:
            self.terminal = True
            return 1
        else:
            return -0.01
    def reset(self):
        self.state = self.initial_state
        self.terminal = False



class Agent:
    def __init__(self,env,gamma,learning_rate,epsilon,num_inputs,num_behaviors):
        self.env = env
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon
        self.num_behaviors = num_behaviors
        self.num_inputs = num_inputs
        self.weights = np.random.randn(self.num_behaviors,self.num_inputs)/np.sqrt(self.num_inputs)
    def s2x(self,s):
        y_normalization_term = Y_REGION//2
        ynt = y_normalization_term
        x_normalization_term = X_REGION//2
        xnt = x_normalization_term
        yr = Y_REGION
        xr = X_REGION

        x =  np.array([

            (s[0] - ynt)/ynt,
            (s[1] - xnt)/xnt,
            (s[0]*s[1] - (xr*yr)//2)/(xr*yr)//2,
            (s[0]*s[0] - (yr*yr)//2)/(yr*yr)//2,
            (s[1]*s[1] - (xr*xr)//2)/(xr*xr)//2,
            (s[0]-self.env.goal_center[0])//X_REGION,
            (s[1]-self.env.goal_center[1])//Y_REGION,
            1

        ])
        #print('x',x)
        x = x.reshape(x.shape[0],1)
        return x
    def forward(self,s):
        x = self.s2x(s)
        return np.dot(self.weights,x)

    def backprop(self,s):
        x = self.s2x(s)
        return x

    def best_behavior_and_value(self,s):
        qs = self.forward(s)
        maxBehavior = np.argmax(qs)
        maxQ = qs[maxBehavior]
        return maxBehavior,maxQ

    def expVSexp(self,behavior):
        p = np.random.random()
        if p < self.eps:
            return np.random.choice(self.num_behaviors)
        else:
            return behavior

    def learn(self,s):
        self.env.state = s

        behavior,value = self.best_behavior_and_value(s)
        behavior = self.expVSexp(behavior)

        prediction = self.forward(s)
        #print('b',behavior)
        value = prediction[behavior]
        #print('prediction',prediction)
        #value = prediction[behavior,0]
        target = np.ones_like(prediction)*prediction

        next_s  = self.env.next_state(behavior)
        best_next_behavior,best_next_behavior_value = self.best_behavior_and_value(next_s)
        reward = self.env.get_reward()
        if self.env.terminal:
            target[behavior,0] = reward
        else:
            target[behavior,0] = reward + self.gamma*best_next_behavior_value
        delta = target - prediction
        #print('t',target,'p',prediction)
        self.weights += self.lr*delta*self.backprop(s).T
        return next_s

env = Environment(Y_REGION,X_REGION,INITIAL_STATE,GOAL)
gamma = 0.9
learning_rate = 0.1
lr_decaying_factor = .1
eps = .5
eps_decaying_factor = 0.5
min_eps = 0.01
num_inputs = 8
num_behaviors = 4
agent = Agent(env,gamma,learning_rate,eps,num_inputs,num_behaviors)

num_episodes = 10000
avg = 0
beta  = 0.9
value = None
it = 1
def exp_mov_avg(beta,it,avg,value):
    m = beta*avg + (1 - beta)*value
    m_hat = m/(1-beta**it)
    return m_hat
episodes = []
for ep in range(num_episodes):
    env.reset()
    s = env.initial_state
    episode_lenght = 0
    episode = [s]
    while not env.terminal:
    #    print('s',s)
        s = agent.learn(s)
        episode.append(s)
        episode_lenght +=1
    #sleep(1)
    episodes.append(episode)
    if ep%(num_episodes//100)==(num_episodes//100)-1:
        print('ep',ep)
        print('LAST EPISODE',episodes[-1])
        print('LENGHT',episode_lenght)
        print('EPSILON',agent.eps)
        #avg,beta = exp_mov_avg(beta,it,avg,value)
        if agent.eps>min_eps:
            agent.eps *=eps_decaying_factor
        agent.lr *=lr_decaying_factor
