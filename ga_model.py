import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
from envs import make_env

def step(env, *args):
    state, a, b, c = env.step(*args)
    state = convert_state(state)
    #state = state
    return state, a, b, c

def reset(env):
    #return env.reset()
    return convert_state(env.reset())

def convert_state(state):
    import cv2
    return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84)) / 255.0
    #return cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (64, 64)) / 255.0

class Model(nn.Module):
    def __init__(self, rng_state):
        super().__init__()
        
        # TODO: padding?
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.dense = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, 18)
        
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
        x = F.relu(self.conv3(x))
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
    print('random_state()')
    return random.randint(0, 2**31-1)

class CompressedModel:
    def __init__(self, start_rng=None, other_rng=None):
        self.start_rng = start_rng if start_rng is not None else random_state()
        self.other_rng = other_rng if other_rng is not None else []
        
    def evolve(self, sigma, rng_state=None):
        self.other_rng.append((sigma, rng_state if rng_state is not None else random_state()))
        
def evaluate_model(env, model, max_eval=20000, max_noop=30,env_seed=2018, render=False, cuda=False):
    import gym
    env = gym.make(env)
    env.seed(env_seed)
    if render: env.render()

    if isinstance(model, CompressedModel):
        model = uncompress_model(model)
    if cuda:
        model.cuda()
    noops = random.randint(0, max_noop)
    cur_states = [reset(env)] * 4
    total_reward = 0
    for _ in range(noops):
        cur_states.pop(0)
        new_state, reward, is_done, _ = step(env, 0)
        total_reward += reward
        if is_done:
            return total_reward
        cur_states.append(new_state)

    total_frames = 0
    model.eval()
    for _ in range(max_eval):
        total_frames += 4
        cur_state_var = Variable(torch.Tensor([cur_states]))
        if cuda:
            cur_state_var = cur_state_var.cuda()
        values = model(cur_state_var)[0]
        action = np.argmax(values.cpu().data.numpy()[:env.action_space.n])
        new_state, reward, is_done, _ = step(env, action)
        total_reward += reward
        if is_done:
            break
        cur_states.pop(0)
        cur_states.append(new_state)
        if render: env.render()


    #print('\t', total_reward)
    return total_reward, total_frames

def evaluate_model_clipped(env, model, max_eval=20000, env_seed=2018, render=False, cuda=False):
    env = make_env(env,env_seed,0,None)()
    if isinstance(model, CompressedModel):
        model = uncompress_model(model)
    if cuda:
        model.cuda()
    obs_shape = env.observation_space.shape
    obs_shape = (obs_shape[0] * 4, *obs_shape[1:])
    current_obs = torch.zeros(1, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = env.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        current_obs[:,:-shape_dim0] = current_obs[:,shape_dim0:]
        current_obs[:,-shape_dim0:] = obs

    if render: env.render()

    obs = env.reset()
    update_current_obs(obs)

    total_frames = 0
    model.eval()
    total_reward = 0.0
    for _ in range(max_eval):
        total_frames += 4
        cur_state_var = Variable(current_obs)
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
        update_current_obs(new_state)
        if render: env.render()

    return total_reward, total_frames

if __name__ == '__main__':
    model = Model(rng_state=1)
    print(evaluate_model('FrostbiteNoFrameskip-v4',model))
