import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

def step(env, *args):
    state, a, b, c = env.step(*args)
    state = convert_state(state)
    return state, a, b, c

def reset(env):
    return convert_state(env.reset())

def convert_state(state):
    import cv2
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84)) / 255.0

class Model(nn.Module):
    def __init__(self, rng_state):
        super().__init__()
        
        # TODO: padding?
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        #self.conv3 = nn.Conv2d(64, 64, (3, 3), 1)
        self.dense = nn.Linear(32 * 9 * 9, 256)
        self.out = nn.Linear(256, 18)
        
        self.rng_state = rng_state
        torch.manual_seed(rng_state)
            
        self.evolve_states = []
            
        self.add_tensors = {}
        for name, tensor in self.named_parameters():
            if tensor.size() not in self.add_tensors:
                self.add_tensors[tensor.size()] = torch.Tensor(tensor.size())
            if 'weight' in name:
                nn.init.kaiming_normal(tensor)
            else:
                tensor.data.zero_()
                        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense(x))
        return self.out(x)
    
    def evolve(self, sigma, rng_state):
        torch.manual_seed(rng_state)
        self.evolve_states.append((sigma, rng_state))
            
        for name, tensor in sorted(self.named_parameters()):
            to_add = self.add_tensors[tensor.size()]
            to_add.normal_(0.0, sigma)
            tensor.data.add_(to_add)
            
    def compress(self):
        return CompressedModel(self.rng_state, self.evolve_states)

def uncompress_model(model):    
    start_rng, other_rng = model.start_rng, model.other_rng
    m = Model(start_rng)
    for sigma, rng in other_rng:
        m.evolve(sigma, rng)
    return m

def random_state():
    return random.randint(0, 2**31-1)

class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []
        
    def evolve(self, sigma, rng_state=None):
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))
        
def evaluate_model(env, model, max_eval=20000, max_noop=30, render=False, cuda=False):
    import gym
    env = gym.make(env)
    model = uncompress_model(model)
    if cuda:
        model.cuda()
    noops = random.randint(0, max_noop)
    cur_states = [reset(env)] * 4
    total_reward = 0
    if render: env.render()
    for _ in range(noops):
        cur_states.pop(0)
        new_state, reward, is_done, _ = step(env, 0)
        total_reward += reward
        if is_done:
            return total_reward
        cur_states.append(new_state)
        if render: env.render()

    total_frames = 0
    model.eval()
    for _ in range(max_eval):
        total_frames += 4
        cur_state_var = Variable(torch.Tensor([cur_states]))
        if cuda:
            cur_state_var = cur_state_var.cuda()
        values = model(cur_state_var)[0]
        if cuda:
            values = values.cpu()
        action = np.argmax(values.data.numpy()[:env.action_space.n])
        new_state, reward, is_done, _ = step(env, action)
        total_reward += reward
        if is_done:
            break
        cur_states.pop(0)
        cur_states.append(new_state)
        if render: env.render()
    if render: env.render(close=True)

    return total_reward, total_frames
