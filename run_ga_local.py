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
parser.add_argument('--total-frames', type=int, default=10000000,
        help='Total frames to play (default: 10000000)')
parser.add_argument('--population', type=int, default=50,
        help='Population of GA (default: 50)')
parser.add_argument('--max-eval', type=int, default=5000,
        help='Max evaluation step per evaluation (default: 5000)')

parser.add_argument('--seed', type=int, default=2018,
        help='Random seed for GA (default: 2018)')
parser.add_argument('--env-name', default='FrostbiteDeterministic-v4',
        help='environment to train on (default: FrostbiteDeterministic-v4)')
parser.add_argument('--save-interval', type=int, default=1,
        help='save interval. (default: 1)')
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
ga = GA(args.population, cuda=args.cuda, seed=args.seed)

def save_model(model, it=None):
    if isinstance(model, ga_model.CompressedModel):
        model = ga_model.uncompress_model(model)
    if args.cuda:
        model = model.cpu()
    save_path = args.save_dir
    try:
        os.makedirs(save_path)
    except OSError:
        pass
    if it:
        filename = os.path.join(save_path, args.env_name+'_{}.pt'.format(it))
    else:
        filename = os.path.join(save_path, args.env_name+'.pt')
    torch.save(model,filename )

# In[ ]:

viswin = None
start=time.time()
elapsed_frames = 0



for it in range(args.total_frames):
    median_score, mean_score, max_score, used_frames, best = ga.evolve_iter(args.env_name,
            max_eval=args.max_eval)
    elapsed_frames += used_frames
    print("Gen {}  Frames {}\tmax:{:.2f}  median:{:.2f}  mean:{:.2f}\ttime:{:.4f}".format(it,
        elapsed_frames, max_score,median_score,mean_score,time.time()-start))
    x = np.column_stack((np.arange(it,it+1), np.arange(it,it+1), np.arange(it,it+1)))
    y = np.array([[max_score, median_score,mean_score]])
    if viswin is None:
        viswin = vis.line(X=x,Y=y,opts=dict(lenged=['max','median','mean']))
    else:
        vis.line(X=x,Y=y,win=viswin,update='append',opts=dict(lenged=['max','median','mean']))
    if it % args.save_interval == 0:
        save_model(best,it=it)
    if elapsed_frames > args.total_frames:
        break

print('Best model saved in {}'.format( os.path.join(save_path, args.env_name+'.pt')))
