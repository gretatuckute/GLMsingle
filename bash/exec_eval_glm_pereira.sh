#!/bin/sh
uids="288"
preprocs="swr"
pcstops="0 1 2 3 4 5 6 7 8 9 10"
fracs="0.4 0.5 0.6 0.7 0.8 0.9 1.0"
resultdir="/om5/group/evlab/u/gretatu/GLMsingle/output_glmsingle"
for uid in $uids ; do
  for preproc in $preprocs ; do
    for pcstop in $pcstops ; do
      for frac in $fracs ; do
         # check whether the file exists! if not, submit sbatch job
          expected_file="output_glmsingle_preproc-${preproc}_pcstop-${pcstop}_fracs-${frac}_UID-${uid}/TYPED_FITHRF_GLMDENOISE_RR.hdf5"
          if [ ! -f "${resultdir}/${expected_file}" ]; then
              echo -e "NOT EXISTING: ${expected_file}\n*********** Submitting sbatch job ***********"
              echo "eval_glm.sh $preproc $pcstop $frac $uid"
              sbatch eval_glm_pereira.sh $preproc $pcstop $frac $uid
          else
              echo -e "EXISTING: $expected_file\n*********** Skipping sbatch job ***********"
          fi
      done
    done
  done
done