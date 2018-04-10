#!/bin/bash
if [ $# -eq 0 ] 
then 
    seed=2018
    device_num=0
else
    seed=$1
    device_num=$2
fi
frames=20000000
pop=100
for i in {Frostbite,Sequest,Pong,SpaceInvaders,BemaRider}; do
    echo CUDA_VISIBLE_DEVICES=${device_num} python run_ga_local.py --env-name "${i}Deterministic-v4" --total-frames ${frames} --population ${pop} --seed ${seed} --save-dir ./trained_models/frames${frames}_seed${seed}/ga/;
    CUDA_VISIBLE_DEVICES=${device_num} python -W ignore run_ga_local.py --env-name "${i}Deterministic-v4" --total-frames ${frames} --population ${pop} --seed ${seed}  --save-dir ./trained_models/frames${frames}_seed${seed}/ga/;
done
