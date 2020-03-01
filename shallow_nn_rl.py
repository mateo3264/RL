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


def ReLU(z):
    z[z<0] = 0
    return z
def ReLU_prime(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z
def get_center(ranges):
    center =(ranges[0][0] + ranges[0][1])/2,(ranges[1][0] + ranges[1][1])/2
    return center
def get_euclidean_distance_to_goal(state,goal):
    s = state
    g = goal
    x_dist = np.abs(s[1]-g[1])
    y_dist = np.abs(s[0]-g[0])
    euc_dis = np.sqrt(np.power(x_dist,2)+np.power(y_dist,2))
    return euc_dis
class Environment:
    def __init__(self,y_region,x_region,initial_state,goal):
        self.y_region = y_region
        self.x_region = x_region
        self.initial_state = initial_state
        self.state = initial_state
        self.goal = goal
        self.goal_center = get_center(self.goal)
        self.terminal = False
        self.last_state = None
        self.last_distance_to_goal = get_euclidean_distance_to_goal(self.state,self.goal_center)
        self.behavior_step = 1
        #self.magnitud_of_movement =
    def next_state(self,behavior,value):
        self.last_state = self.state
        y = 0
        x = 0
        #print('v',value)
        if behavior == 0:
            y =-self.behavior_step
        elif behavior == 1:
            x =-self.behavior_step
        elif behavior == 2:
            y = self.behavior_step
        elif behavior == 3:
            x = self.behavior_step
        if 0 <= self.state[1] + x <= X_REGION and 0 <= self.state[0] + y <= Y_REGION:
            #sección para colocar código que verifique si el agente esta dentro de
            #una región de un bloque
            self.state = (self.state[0]+y ,self.state[1]+x)
        return self.state

    def get_reward(self):
        new_dist = get_euclidean_distance_to_goal(self.state,self.goal_center)
        if self.goal[0][0]<=self.state[0]<=self.goal[0][1] and \
            self.goal[1][0]<=self.state[1]<=self.goal[1][1]:
            self.terminal = True
            return 1

#        elif new_dist<self.last_distance_to_goal:
#            return 1/new_dist
        else:
            return -0.01
    def reset_goal(self):
        min_y = np.random.uniform(0,Y_REGION-self.behavior_step)
        max_y = min_y + 1
        min_x = np.random.uniform(0,X_REGION-self.behavior_step)
        max_x = min_x + 1
        new_goal_y = (min_y,max_y)
        new_goal_x = (min_x,max_x)
        self.goal = (new_goal_y,new_goal_x)
        self.goal_center = get_center(self.goal)

    def reset(self):
        self.state = self.initial_state
        self.terminal = False
        #self.reset_goal()



class Agent:
    def __init__(self,env,gamma,learning_rate,epsilon,layers):
        self.env = env
        self.gamma = gamma
        self.lr = learning_rate
        self.eps = epsilon
        self.num_behaviors = num_behaviors
        self.num_inputs = num_inputs
        self.layers = layers
        self.num_layers = len(self.layers)
        self.weights = [np.random.randn(y,x)/np.sqrt(x) for
                        x,y in zip(self.layers[:-1],self.layers[1:])
                        ]
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
        a = self.s2x(s)
        for l in range(self.num_layers-1):
            z = np.dot(self.weights[l],a)
            if l == self.num_layers -2:
                a = z
            else:
                a = ReLU(z)
        return a

    def get_td_error(self,s):
        behavior,value = self.best_behavior_and_value(s)
        behavior = self.expVSexp(behavior)
        prediction = self.forward(s)
        value = prediction[behavior][0]
        target = np.ones_like(prediction)*prediction

        next_s  = self.env.next_state(behavior,value)
        best_next_behavior,best_next_behavior_value = self.best_behavior_and_value(next_s)
        reward = self.env.get_reward()
        if self.env.terminal:
            target[behavior,0] = reward
        else:
            target[behavior,0] = reward + self.gamma*best_next_behavior_value
        delta = target - prediction
        return delta,next_s

    def backprop(self,s):
        grads = []
        x = self.s2x(s)
        z1 = np.dot(self.weights[0],x)
        a1 = ReLU(z1)
        z2 = np.dot(self.weights[1],a1)
        a2 = z2

        delta,next_s = self.get_td_error(s)
        dw2 = np.dot(delta,a1.T)
        da1 = np.dot(self.weights[1].T,delta)

        delta = da1*ReLU_prime(z1)
        dw1 = np.dot(delta,x.T)
        grads.append(dw2)
        grads.append(dw1)

        return grads,next_s

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
        grads,next_s = self.backprop(s)
        #print('t',target,'p',prediction)
        for l,grad in enumerate(grads):
            self.weights[-l-1] += self.lr*grad
        return next_s

env = Environment(Y_REGION,X_REGION,INITIAL_STATE,GOAL)
gamma = 0.9
learning_rate = .01
lr_decaying_factor = .9
eps = 0.5
eps_decaying_factor = 0.9
min_eps = 0.01
num_inputs = 8
num_behaviors = 4
layers = [num_inputs,100,num_behaviors]
agent = Agent(env,gamma,learning_rate,eps,layers)

num_episodes = 10000
avg = 0
beta  = 0.9
value = None
it = 1

def print_hiperparameters(gamma,learning_rate):
    print('GAMMA:',gamma)
    print('LEARNING RATE',learning_rate)
print_hiperparameters(gamma,learning_rate)

def exp_mov_avg(beta,it,avg,value):
    m = beta*avg + (1 - beta)*value
    m_hat = m/(1-beta**it)
    return m_hat
episodes = []
for ep in range(num_episodes):
    env.reset()
    print('GOAL',env.goal_center)
    s = env.initial_state
    episode_lenght = 0
    episode = [s]
    while not env.terminal:
    #    print('s',s)
        s = agent.learn(s)
        episode.append(s)
        episode_lenght +=1
    #sleep(1)
    #print('episode',episode)
    #print('length',episode_lenght)
    episodes.append(episode_lenght)
    if ep%(num_episodes//1000)==(num_episodes//1000)-1:
        print('ep',ep)
        if episode_lenght<=50:
            print('LAST EPISODE',episodes[-1])
        print('LENGHT',episode_lenght)
        print('EPSILON',agent.eps)

        #avg,beta = exp_mov_avg(beta,it,avg,value)
        if agent.eps>min_eps:
            agent.eps *=eps_decaying_factor
        agent.lr *=lr_decaying_factor

    print('GOAL',env.goal_center,episodes[-1])
