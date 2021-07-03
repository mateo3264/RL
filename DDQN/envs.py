import numpy as np
from PIL import Image
import cv2


class ActionSpace:
    def __init__(self):
        self.n = 4
        self.ini()
    def ini(self):
        return [a for a in range(self.n)]
      
      
class Env:
    def __init__(self,rows,cols,walls,terminal_states):
        self.rows = rows
        self.cols = cols
        self.walls = walls
        self.observation_space = np.zeros((self.rows,self.cols))
        self.terminal_states = terminal_states
        self.initial_state = (self.rows - 1,0)
        self.current_state = self.initial_state
        self.action_space = ActionSpace()

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
        return np.array(self.initial_state)


class ShmupEnv:
    def __init__(self,rows,cols,n_mobs):
        self.rows = rows
        self.cols = cols
        self.n_mobs = n_mobs
        self.initial_agent_position = [self.rows-1,self.cols//2]
        self.current_agent_pos = self.initial_agent_position

        self.mob_poss = self.spawn_mobs()
        self.initial_pos = self.get_state()
        #self.
        self.current_state = self.initial_pos
        self.mob_poss = self.spawn_mobs()
        
    def reset(self):
        self.initial_agent_position = [self.rows-1,self.cols//2]
        self.current_agent_pos = self.initial_agent_position

        self.mob_poss = self.spawn_mobs()
        self.initial_pos = self.get_state()
        self.current_state = self.initial_pos
        
        return self.current_state

    def get_state(self):
        state = []
        for mob_pos in self.mob_poss:
            for mob_coor in mob_pos:
                state.append(mob_coor)
        for agent_coor in self.current_agent_pos:
            state.append(agent_coor)
        return state

            
    def spawn_mobs(self):
        mob_poss = []
        for i in range(self.n_mobs):
            row = 0#np.random.randint(0,0)
            col = np.random.randint(self.cols)
            mob_poss.append([row,col])
        return mob_poss

    def spawn_one_mob(self,mob_idx):
        row = 0
        col = (self.current_agent_pos[1]+ np.random.randint(-2,3))%self.cols
        self.mob_poss[mob_idx] = [row,col]
        
    def calculate_dist(self):
        if self.cols/2 - self.current_agent_pos[1] > 0:
            return np.abs(-1-self.current_agent_pos[1])
        return np.abs(self.cols-self.current_agent_pos[1])
    def step(self,action):
        r,c = self.current_agent_pos
        if action == 2:
            r -=1
        if action == 0:
            c -=1
        if action == 3:
            r +=1
        if action == 1:
            c +=1
        


        if r < 0:
            r = 0
        elif r > self.rows - 1:
            r = self.rows - 1
        elif c < 0:
            c = 0
        elif c > self.cols - 1:
            c = self.cols - 1
        self.current_agent_pos = (r,c)
        reward = 1 - (5/self.calculate_dist())
        #print('reward',reward)
        done = False
        
        for i in range(self.n_mobs):
            mob_pos = self.mob_poss[i][0]
            mob_pos += 1
            if mob_pos > self.rows - 1:
                self.spawn_one_mob(i)
            else:
                self.mob_poss[i][0] +=1
        self.current_state = self.get_state()
        for i in range(self.n_mobs):

            if (r,c) == tuple(self.mob_poss[i]):
                done = True
                reward = -10
                return self.current_state,reward,done
        return self.current_state,reward,done#tuple(mob_coor for mob_coor in mob_pos for mob_pos in self.mob_poss)+(r,c),reward,done

    def render(self):
        z = np.zeros((self.rows,self.cols,3),dtype=np.int8)

        z[self.current_agent_pos] = [255,0,0]
        near = False
        ticks = 1
        for mob_pos in self.mob_poss:
            z[mob_pos[0],mob_pos[1]] = [0,0,255]
            if mob_pos[0]>9:
                near = True
        if near:
            ticks = 200
        img = Image.fromarray(z,'RGB')
        img = img.resize((250,250),Image.NEAREST)
        
        cv2.imshow('',np.array(img))
        cv2.waitKey(1)

    def state_as_pixels(self,state=None):
            z = np.zeros((self.rows,self.cols),dtype=np.int8)

            z[self.current_agent_pos] = 2

            
            for mob_pos in self.mob_poss:
                z[mob_pos[0],mob_pos[1]] = -1

        
            z = z.flatten()
            return z

    def states_as_pixels(self,states):
        pass


if __name__ == '__main__':
    env = Env(3,4,[(1,1)],{(0,3):1,(1,3):-1})


