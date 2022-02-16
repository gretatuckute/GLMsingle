#!/bin/sh
uids="18"
preprocs="swr"
pcstops="1 3 5 7 9"
fracs="0.5 0.7 0.9"
for uid in $uids ; do
  for preproc in $preprocs ; do
    for pcstop in $pcstops ; do
      for frac in $fracs ; do
          echo "eval_glm.sh $preproc $pcstop $frac $uid"
          sbatch eval_glm_pereira.sh $preproc $pcstop $frac $uid
      done
    done
  done
done