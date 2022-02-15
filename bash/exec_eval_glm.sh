#!/bin/sh
preprocs="swr"
pcstops="1 2 3 4 5 6 7 8 9 10"
fracs="0.5 0.6 0.7 0.8 0.9 0.95 1.0"
for preproc in $preprocs ; do
  for pcstop in $pcstops ; do
    for frac in $fracs ; do
        echo "eval_glm.sh $preproc $pcstop $frac"
        sbatch eval_glm.sh $preproc $pcstop $frac
    done
  done
done