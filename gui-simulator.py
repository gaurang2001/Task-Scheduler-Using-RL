import tkinter
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import random
from tensorflow import keras
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

def display(arr, reww, flag):
    l = []
    if flag:
        rew.set(reww)
        for i in range(5):
            if arr[i][0]==1:
                l.append(i+1)

        if len(l)!=0:
            task_msg.set(', '.join(map(str, l)))
        else:
            task_msg.set("No tasks were scheduled.")

    else:
        rl_rew.set(reww)
        for i in range(5):
            if arr[i][0]==1:
                l.append(i+1)

        if len(l)!=0:
            rl_task.set(', '.join(map(str, l)))
        else:
            rl_task.set("No tasks were scheduled.")

def build_agent(model, actions):
    policy = EpsGreedyQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=200, target_model_update=1e-2)
    return dqn


class TaskEnv(Env):
    def __init__(self, resources=[90, 90], num_tasks=5):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(num_tasks)
        # Temperature array
        self.observation_space = Box(low=0, high=90, shape=(num_tasks, len(resources)))
        # Set state
        self.state_ = np.zeros((num_tasks, 1), dtype=int)
        self.limit_ = 0
        for lim in resources:
            self.state_ = np.append(self.state_, np.random.randint(lim, size=(num_tasks, 1)), axis=1)
            
        self.reward_ = 0
        for lim in resources:
            self.limit_+=lim   #Set the total resource limit as sum of resources
       
        self.num_tasks_ = num_tasks
        self.resources_ = resources
        
    def getSum(self, x ):
        if x[0] == 1:
            return sum(x)-1
        else:
            return 0    
    
    def step(self, actionIdx):
        ### update reward
        self.reward_ = 0
        reward = 0
        done = False
        #update the state based on action
        #case 1, if same task selected then penalize the agent
        if self.state_[actionIdx][0] == 1:
          # get the sum for selected task
            sum_res = np.sum(self.state_[actionIdx], axis=0)
            reward = -sum_res/10
            return self.state_,reward,done,{}
        else:
            self.state_[actionIdx][0] = 1
       
        #collect all the resources for this batch until now.
        totReward = sum(np.apply_along_axis( self.getSum, axis=1, arr=self.state_ ))
        # get the sum for selected task
        sum_res = np.sum(self.state_[actionIdx], axis=0) 
        ones = np.sum(self.state_, axis=0)
        if totReward <= self.limit_:
            if ones[0] >= 5:
              done = True
            reward = sum_res
        else:
            reward = -sum_res/10
            done = True
            
        return self.state_,reward,done,{}

    def render(self):
        # Implement viz
        print(self.state_)
    
    def set_state(self, state):
        self.reset()
        self.state_ = state

    def reset(self):
        state = np.zeros((self.num_tasks_, 1), dtype=int)
        for lim in self.resources_: 
            state = np.append(state, np.random.randint(lim, size=(self.num_tasks_,1)), axis=1)
        self.state_ = state
        self.reward_ = 0
        return self.state_

def fcfs(state):
    env = TaskEnv()
    env.set_state(state)
    done = False
    score = 0 
    i = 0
    
    while not done:
        #env.render()
        action = i
        i = i+1
        state, reward, done, info = env.step(action)
        # print(n_state)
        # print(action, " ", dqn.forward(n_state))
        score+=reward
    return score, state

def process(state):
    env = TaskEnv()
    model = keras.models.load_model('bestmodelyet')
    actions = env.action_space.n
    dqn = build_agent(model, actions)
    env.set_state(state)
    done = False
    score = 0
    i = 0
    while not done:
        action = dqn.forward(state)
        state, reward, done, info = env.step(action)
        i += 1
        if i > 100:
            break
        score+=reward
    return score, state

def send(event=None):
    """ Handles sending of messages. """
    msg = my_msg.get()

    if msg == "-1":
        top.quit()
        return

    res0 = [0]*5
    res1 = [float(ef1.get()),float(ef2.get()),float(ef3.get()),float(ef4.get()),float(ef5.get())]
    res2 = [float(eff1.get()),float(eff2.get()),float(eff3.get()),float(eff4.get()),float(eff5.get())]
    res = np.stack((res0,res1,res2),axis=1)
    res_copy = res.copy()
    score, state = fcfs(res)
    display(state, score, True)

    score, state = process(res_copy)

    display(state, score, False)

def on_closing(event=None):
    """ This function is to be called when the window is closed. """
    my_msg.set(-1)
    send()

top = tkinter.Tk()
top.title("Task Scheduling using RL")
messages_frame = tkinter.Frame(top)

n1 = tkinter.StringVar()
n2 = tkinter.StringVar()
n3 = tkinter.StringVar()
n4 = tkinter.StringVar()
n5 = tkinter.StringVar()

nn1 = tkinter.StringVar()
nn2 = tkinter.StringVar()
nn3 = tkinter.StringVar()
nn4 = tkinter.StringVar()
nn5 = tkinter.StringVar()

my_msg = tkinter.StringVar()
my_msg.set("")

task_msg = tkinter.StringVar() 
task_msg.set("")

rew = tkinter.StringVar()
rew.set("")

rl_task = tkinter.StringVar() 
rl_task.set("")

rl_rew = tkinter.StringVar()
rl_rew.set("")

button_label = tkinter.Label(messages_frame, text="\nFCFS Scheduling\n")
button_label.grid(columnspan=5,row=1)

msg_list = tkinter.Label(messages_frame, text="Tasks that were scheduled: ")
msg_list.grid(column=1,row=2)

msg_list1 = tkinter.Label(messages_frame, textvariable=task_msg)
msg_list1.grid(column=2,columnspan=2,row=2)

msg_list2 = tkinter.Label(messages_frame, text="Reward: ")
msg_list2.grid(column=1,row=3)

msg_list3 = tkinter.Label(messages_frame, textvariable=rew)
msg_list3.grid(column=2,columnspan=2,row=3)

button_label = tkinter.Label(messages_frame, text="\nRL Scheduling\n")
button_label.grid(columnspan=5,row=4)

msg_list4 = tkinter.Label(messages_frame, text="Tasks that were scheduled: ")
msg_list4.grid(column=1,row=5)

msg_list5 = tkinter.Label(messages_frame, textvariable=rl_task)
msg_list5.grid(column=2,columnspan=2,row=5)

msg_list6 = tkinter.Label(messages_frame, text="Reward: ")
msg_list6.grid(column=1,row=6)

msg_list7 = tkinter.Label(messages_frame, textvariable=rl_rew)
msg_list7.grid(column=2,columnspan=2,row=6)


messages_frame.grid(columnspan=5)

button_label = tkinter.Label(top, text="Enter CPU & Memory Utilization for 5 Processes\n")
button_label.grid(columnspan=5,row=1)

button_label = tkinter.Label(top, text="")
button_label.grid(column=1,row=2)

button_label = tkinter.Label(top, text="CPU Utilisation")
button_label.grid(column=2,row=2)

button_label = tkinter.Label(top, text="Memory Utilization")
button_label.grid(column=3,row=2)

#################

ef1 = tkinter.Entry(top, textvariable=n1)
ef1.grid(column=2, row=3)
el1 = tkinter.Label(top, text="Process 1: ")
el1.grid(column=1, row=3)

ef2 = tkinter.Entry(top, textvariable=n2)
ef2.grid(column=2, row=4)
el2 = tkinter.Label(top, text="Process 2: ")
el2.grid(column=1, row=4)

ef3 = tkinter.Entry(top, textvariable=n3)
ef3.grid(column=2, row=5)
el3 = tkinter.Label(top, text="Process 3: ")
el3.grid(column=1, row=5)

ef4 = tkinter.Entry(top, textvariable=n4)
ef4.grid(column=2, row=6)
el4 = tkinter.Label(top, text="Process 4: ")
el4.grid(column=1, row=6)

ef5 = tkinter.Entry(top, textvariable=n5)
ef5.grid(column=2, row=7)
el5 = tkinter.Label(top, text="Process 5: ")
el5.grid(column=1, row=7)

#################

eff1 = tkinter.Entry(top, textvariable=nn1)
eff1.grid(column=3, row=3)

eff2 = tkinter.Entry(top, textvariable=nn2)
eff2.grid(column=3, row=4)

eff3 = tkinter.Entry(top, textvariable=nn3)
eff3.grid(column=3, row=5)

eff4 = tkinter.Entry(top, textvariable=nn4)
eff4.grid(column=3, row=6)

eff5 = tkinter.Entry(top, textvariable=nn5)
eff5.grid(column=3, row=7)

button_label = tkinter.Label(top, text="\n")
button_label.grid(columnspan=5,row=8)

send_button = tkinter.Button(top, text="Run", command=send)
send_button.grid(column=2,row=9)
quit_button = tkinter.Button(top, text="Quit", command=on_closing)
quit_button.grid(column=2,row=10)

button_label = tkinter.Label(top, text="\n")
button_label.grid(columnspan=5,row=11)

top.protocol("WM_DELETE_WINDOW", on_closing)

top.mainloop()
