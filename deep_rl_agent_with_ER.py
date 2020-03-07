#implementar er
#-forward con la target nn
#-se está repitiendo mucho el forward
import numpy as np
from time import sleep
from ttkthemes import ThemedTk as tk
from tkinter import ttk
from collections import deque
import random

class Window:
    def __init__(self):
        self.root = tk(theme='radiance')
        #self.root.get_themes()
        #self.root.set_theme('radiance')
        self.b = ttk.Button(self.root,text='hola')
        self.b.pack()
        self.root.mainloop()
#w = Window()

Y_REGION = 16
X_REGION = 20
INITIAL_STATE = (0.5,0.5)
GOAL = ((4.3,5.3),(8.9,9.9))
OBSTACLES = []


def ReLU(z):
    #print('relu',z)
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
        self.theta = [ws.copy() for ws in self.weights]
        self.tao = 0.01
        self.experiences = []
        self.max_num_experiences = 1000

        self.mini_batch_size = 32

    def __str__(self):
        return 'Shallow NN Agent with ER'

    def s2x(self,s):
        y_normalization_term = Y_REGION//2
        ynt = y_normalization_term
        x_normalization_term = X_REGION//2
        xnt = x_normalization_term
        yr = Y_REGION
        xr = X_REGION



        if isinstance(s,tuple):
            m = 1
            sy = s[0]
            sx = s[1]
            x =  np.array([

                (sy - ynt)/ynt,
                (sx - xnt)/xnt,
                (sy*sx - (xr*yr)//2)/(xr*yr)//2,
                (sy*sy - (yr*yr)//2)/(yr*yr)//2,
                (sy*sy - (xr*xr)//2)/(xr*xr)//2,
                (sx-self.env.goal_center[0])//X_REGION,
                (sx-self.env.goal_center[1])//Y_REGION,
                1,

            ])

        else:
            m = s.shape[1]
            sy = s[0]
            sx = s[1]

            x =  np.array([

                list((sy - ynt)/ynt),
                list((sx - xnt)/xnt),
                list((sy*sx - (xr*yr)//2)/(xr*yr)//2),
                list((sy*sy - (yr*yr)//2)/(yr*yr)//2),
                list((sy*sy - (xr*xr)//2)/(xr*xr)//2),
                list((sx-self.env.goal_center[0])//X_REGION),
                list((sx-self.env.goal_center[1])//Y_REGION),
                [1]*m,

            ])
        x = x.reshape(x.shape[0],m)
        return x
    def target_forward(self,s):
        a = self.s2x(s)
        for l in range(self.num_layers-1):
            z = np.dot(self.theta[l],a)
            if l == self.num_layers -2:
                a = z
            else:
                a = ReLU(z)

        return a

    def forward(self,s):
        a = self.s2x(s)
        for l in range(self.num_layers-1):
            z = np.dot(self.weights[l],a)
            if l == self.num_layers -2:
                a = z
            else:
                a = ReLU(z)

        return a

    def target_nn_forward(self,s):
        a = self.s2x(s)
        activations = [a]
        zs = []

        for l in range(self.num_layers-1):
            z = np.dot(self.theta[l],a)
            zs.append(z)
            if l == self.num_layers -2:
                a = z
            else:
                a = ReLU(z)
            activations.append(a)
        return a,activations,zs
    def get_td_error(self,s):
        behavior,value = self.best_behavior_and_value(s)
        behavior = self.expVSexp(behavior)
        prediction = self.forward(s)
        value = prediction[behavior][0]#???
        target = np.ones_like(prediction)*prediction

        next_s  = self.env.next_state(behavior,value)
        best_next_behavior,best_next_behavior_value = self.best_behavior_and_value(next_s)
        reward = self.env.get_reward()
        if self.env.terminal:
            target[behavior,0] = reward
        else:
            target[behavior,0] = reward + self.gamma*best_next_behavior_value
        delta = target - prediction
        self.experiences.append((s,behavior,next_s,reward,self.env.terminal))
        if len(self.experiences)>self.max_num_experiences:
            del self.experiences[0]

        return delta,next_s



    def backprop(self,s):
        grads = []
        x = self.s2x(s)
        a = x
        zs = []
        activations = [a]
        for l in range(self.num_layers-1):
            z = np.dot(self.weights[l],a)
            zs.append(z)
            if l == self.num_layers -2:
                a = z
            else:
                a = ReLU(z)
            activations.append(a)


        delta,next_s = self.get_td_error(s)
        dw = np.dot(delta,activations[-2].T)
        da = np.dot(self.weights[-1].T,delta)

        grads.append(dw)

        for l in range(2,self.num_layers):
            cached_a = activations[-l-1]
            cached_z = zs[-l]
            delta = da*ReLU_prime(cached_z)
            dw = np.dot(delta,cached_a.T)
            grads.append(dw)
            da = np.dot(self.weights[-l].T,delta)

        return grads,next_s

    def unpack_experiences(self,mini_batch):

        tmp_list = np.array(mini_batch).T
        ss,bs,next_ss,rs,dones = tmp_list
        ss,next_ss = list(ss),list(next_ss)
        ss,next_ss = np.array(ss).T,np.array(next_ss).T
        bs = bs.reshape(1,bs.shape[0])
        rs = rs.reshape(1,rs.shape[0])
        dones = dones.reshape(1,dones.shape[0])
        return ss,bs,next_ss,rs,dones

    def get_er_td_error(self,mini_batch):
        s_ms,b_ms,next_s_ms,r_ms,done_ms = self.unpack_experiences(mini_batch)
        q_ms = self.forward(s_ms)
        target_ms = q_ms.copy()

        next_b_ms,next_q_ms = self.best_behavior_and_value(next_s_ms,target_nn=True)

        next_q_ms = self.forward(next_s_ms)[next_b_ms,[x for x in range(len(mini_batch))]]

        non_terminal_target_ms = r_ms[0] + self.gamma*next_q_ms - q_ms
        terminal_target_ms = r_ms

#        try:
        respective_targets = [terminal_target_ms[0,i] if done_ms[0,i] else non_terminal_target_ms[0,i] for i in range(len(done_ms[0]))]
#        except:
#            print('EXCEPT',done_ms)
#            respective_targets = [terminal_target_ms[i] if done_ms[0,i] else non_terminal_target_ms[i] for i in range(len(done_ms))]

        cols = [x for x in range(len(mini_batch))]

        target_ms[b_ms[0].astype('int'),np.array(cols)] = respective_targets
        delta_ms = target_ms - q_ms
        return delta_ms


    def er_backprop(self):
        grads = []
        if len(self.experiences) < self.mini_batch_size:
            k = len(self.experiences)
        else:
            k = self.mini_batch_size
        mini_batch = random.sample(self.experiences,k)
        s_ms,b_ms,next_s_ms,r_ms,done_ms = self.unpack_experiences(mini_batch)
        x_ms = self.s2x(s_ms)
        a_ms = x_ms
        zs = []
        activations = [a_ms]
        for l in range(self.num_layers-1):
                z_ms = np.dot(self.weights[l],a_ms)
                zs.append(z_ms)
                if l == self.num_layers -2:
                    a_ms = z_ms
                else:
                    a_ms = ReLU(z_ms)
                activations.append(a_ms)


        delta_ms = self.get_er_td_error(mini_batch)
        dw = np.dot(delta_ms,activations[-2].T)

        da_ms = np.dot(self.weights[-1].T,delta_ms)

        grads.append(dw)

        for l in range(2,self.num_layers):
            cached_a_ms = activations[-l-1]
            cached_z_ms = zs[-l]
            delta_ms = da_ms*ReLU_prime(cached_z_ms)
            dw = np.dot(delta_ms,cached_a_ms.T)
            dw = (1/self.mini_batch_size)*dw
            grads.append(dw)
            da_ms = np.dot(self.weights[-l].T,delta_ms)

        self.train_with_experiences(grads)
        #Double DQN
        tmp_ws = [ws.copy() for ws in self.weights]
        tmp_thetas = [ts.copy() for ts in self.theta]
        #print('BEFORE',type(self.theta))

        self.theta = [self.theta[l] + self.tao*tmp_ws + (1 - self.tao)*tmp_thetas
                        for l,(tmp_ws,tmp_thetas) in enumerate(zip(tmp_ws,tmp_thetas))
                        ]
        #print('AFTER',type(self.theta))
    def train_with_experiences(self,grads):
        for l,dw in enumerate(grads):
            self.weights[-l-1] += self.lr*dw

    def experience_replay(self):
        self.er_backprop()

    def best_behavior_and_value(self,s,target_nn=False):
        if target_nn:
            qs = self.target_forward(s)
        else:
            qs = self.forward(s)

        try:
            x_axis = qs.shape[1]
            maxBehavior = np.argmax(qs,axis=0)
            maxQ = qs[maxBehavior,[x for x in range(x_axis)]]
        except:
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

        for l,grad in enumerate(grads):
            self.weights[-l-1] += self.lr*grad

        self.experience_replay()


        return next_s

env = Environment(Y_REGION,X_REGION,INITIAL_STATE,GOAL)
gamma = 0.9
learning_rate = .0001
lr_decaying_factor = .9
eps = .2
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

def print_hiperparameters(gamma,learning_rate,eps):
    print('GAMMA:',gamma)
    print('LEARNING RATE',learning_rate)
    print('EPSILON',eps)
    print('AGENT',agent)
print_hiperparameters(gamma,learning_rate,eps)

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
