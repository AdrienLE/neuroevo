#!/bin/bash
seed=2018
frames=30000000
pop=100
for i in {Frostbite,Sequest,Pong,SpaceInvaders,BemaRider}; do
    echo python run_ga_local.py --env-name "${i}Deterministic-v4" --total-frames ${frames} --population ${pop} --save-dir ./trained_models/frames${frames}_seed${seed}/ga/;
    python -W ignore run_ga_local.py --env-name "${i}Deterministic-v4" --total-frames ${frames} --population ${pop} --save-dir ./trained_models/frames${frames}_seed${seed}/ga/;
done
