import numpy as np

class Env:
    def __init__(self,rows,cols,walls,terminal_states):
        self.rows = rows
        self.cols = cols
        self.walls = walls
        self.terminal_states = terminal_states
        self.initial_state = (self.rows - 1,0)
        self.current_state = self.initial_state
    def step(self,action):
        r,c = self.current_state
        if action == 0:
            r -=1
        elif action == 1:
            c -=1
        elif action == 2:
            r +=1
        elif action == 3:
            c +=1

        reward = -0.01
        done = False

        if 0 <= r <= self.rows - 1 and 0 <= c <= self.cols - 1:
            if (r,c) not in self.walls:
                if (r,c) in self.terminal_states:
                    reward = self.terminal_states[(r,c)]
                    done = True
                self.current_state = (r,c)
        return self.current_state,reward,done
    def reset(self):
        self.initial_state = (self.rows - 1,0)
        self.current_state = self.initial_state
        return self.initial_state



class Agent:
    def __init__(self,env,alpha=0.1,epsilon=0.1,om=0.9):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.om = om
        self.Q = np.zeros((env.rows,env.cols,4))
        self.steps = 0
        self.states_actions = []
        self.D = 0
        self.K = 1
        self.idx = 0
        self.n_times_actions_taken = np.zeros((env.rows,env.cols,4))
    def hyper(self,D):
        return 1/(1 + self.K*D)
        
    def choose_action(self,s):
        p = np.random.rand()
        if p < self.epsilon:
            return np.random.choice(4)
        else:
            return np.argmax(self.Q[s[0],s[1]])
    def expected_sarsa(self,s_):
        next_q = 0
        for a in range(4):
            next_q += (self.n_times_actions_taken[s_[0],s_[1],a]/np.sum(self.n_times_actions_taken[s_[0],s_[1]]))*self.Q[s_[0],s_[1],a]
        return next_q
    def learn(self,s,a,s_,reward,done):
        if done:
            target = reward
        else:
            if self.idx == 1 or self.idx == 3:
                target = reward + self.om*self.expected_sarsa(s_)
            elif self.idx == 0 or self.idx == 2:
                target = reward + self.om*np.max(self.Q[s_[0],s_[1]])
        if self.idx == 0 or self.idx == 3:
            for step in range(self.steps):
                past_s,past_a = self.states_actions[step]

                h = self.hyper(self.D - step)
                self.Q[past_s[0],past_s[1],past_a] += h*self.alpha*(target - self.Q[past_s[0],past_s[1],past_a])
        elif self.idx == 1:
            self.Q[s[0],s[1],a] += self.alpha*(target - self.Q[s[0],s[1],a])
        else:

            self.Q[s[0],s[1],a] += self.alpha*(target - self.Q[s[0],s[1],a])
episodes = 100
rows = 20
cols = 30
env = Env(rows,cols,[(1,1)],{(0,cols-1):1,(1,cols-1):-1})
agent = Agent(env,0.1,0.1,0.9)
experiments = 4
stepss = [[] for i in range(experiments)]

for i in range(experiments):
    agent.Q = np.zeros_like(agent.Q)
    for e in range(episodes):
        done = False
        s = env.reset()
        agent.steps = 0
        agent.D = 0
        agent.states_actions = []
        agent.idx = i

        while not done:
            a = agent.choose_action(s)
            agent.n_times_actions_taken[s[0],s[1],a] +=1
            s_,reward,done = env.step(a)
            agent.states_actions.append((s,a))
            agent.steps +=1
            agent.learn(s,a,s_,reward,done)
            agent.D += 1
            s = s_
        stepss[i].append(agent.steps)

import matplotlib.pyplot as plt

for i in range(experiments):
    print(stepss[i][:20])

lbls = ['behavioral','expected sarsa','Q-learning','more behavioral']
for i in range(experiments):
    plt.plot([x for x in range(len(stepss[i]))],stepss[i],label=lbls[i])
plt.legend()
plt.ylabel('Numero de pasos para completar tarea')
plt.show()
