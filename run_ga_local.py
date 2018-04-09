# coding: utf-8
import os
import torch
import time
import visdom
import numpy as np
import argparse

import ga_model
from local_ga import GA

parser = argparse.ArgumentParser(description='GA RL')
parser.add_argument('--total-frames', type=int, default=50000000,
        help='Total frames to play (default: 50000000)')
parser.add_argument('--population', type=int, default=10,
        help='Population of GA (default: 10)')
parser.add_argument('--env-name', default='FrostbiteNoFrameskip-v4',
        help='environment to train on (default: FrostbiteNoFrameskip-v4)')
parser.add_argument('--save-dir', default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False,
        help='disables visdom visualization')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.vis = not args.no_vis



vis = visdom.Visdom(env=args.env_name)
ga = GA(args.population, cuda=args.cuda)

# In[ ]:

viswin = None
start=time.time()
elapsed_frames = 0
for it in range(args.total_frames):
    median_score, mean_score, max_score, used_frames = ga.evolve_iter(args.env_name)
    elapsed_frames += used_frames
    print("Gen {}  Frames {}\tmax:{:.2f}  median:{:.2f}  mean:{:.2f}\ttime:{:.4f}".format(it,
        elapsed_frames, max_score,median_score,mean_score,time.time()-start))
    x = np.column_stack((np.arange(it,it+1), np.arange(it,it+1), np.arange(it,it+1)))
    y = np.array([[max_score, median_score,mean_score]])
    if viswin is None:
        viswin = vis.line(X=x,Y=y,opts=dict(lenged=['max','median','mean']))
    else:
        vis.line(X=x,Y=y,win=viswin,update='append',opts=dict(lenged=['max','median','mean']))
    if elapsed_frames > args.total_frames:
        break

scored_models, used_frames = ga.get_best_models(args.env_name)
best = ga_model.uncompress_model(scored_models[0][0])
if args.cuda:
    best = best.cpu()
save_path = args.save_dir
try:
    os.makedirs(save_path)
except OSError:
    pass
torch.save(best, os.path.join(save_path, args.env_name+'.pt'))
print('Best model saved in {}'.format( os.path.join(save_path, args.env_name+'.pt')))
