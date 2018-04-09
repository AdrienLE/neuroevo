#!/bin/bash
seed=2018
frames=50000000
pop=50
for i in {Frostbite,Sequest,Pong,SpaceInvaders,BemaRider}; do
    echo python run_ga_local.py --env-name "${i}NoFrameskip-v4" --total-frames ${frames} --population ${pop} --save-dir ./trained_models/frames${frames}_seed${seed};
    python -W ignore run_ga_local.py --env-name "${i}NoFrameskip-v4" --total-frames ${frames} --population ${pop} --save-dir ./trained_models/frames${frames}_seed${seed};
done
