#!/usr/bin/bash
#
# Usage: ./example.sh

TIMESTEP=5e-4
MEASURETIME=1
SIMTIME=80
TX=-0.5
MODEL=xxz
VX=8.0
TY=-1
VY=4
tevo --model=$MODEL --lattice-size=121,2 --wall-start=60,0 --wall-size=1,2 --gaussian-start=35,0 --gaussian-size=11,1 --gaussian-center=40 --gaussian-sigma=3.0 --tx=$TX --vx=$VX --ty=$TY --vy=$VY --simulation-time=$SIMTIME --time-per-measurement=$MEASURETIME --time-step=$TIMESTEP --verbosity=1
