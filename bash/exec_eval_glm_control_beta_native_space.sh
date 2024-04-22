#!/bin/sh
uids="848"
preprocs="r"
pcstops="5"
fracs="0.05"
resultdir="/nese/mit/group/evlab/u/gretatu/GLMsingle/output_glmsingle_native_space"
for uid in $uids ; do
  for preproc in $preprocs ; do
    for pcstop in $pcstops ; do
      for frac in $fracs ; do
         # check whether the file exists! if not, submit sbatch job
          expected_file="output_glmsingle_preproc-${preproc}_pcstop-${pcstop}_fracs-${frac}_UID-${uid}/TYPED_FITHRF_GLMDENOISE_RR.hdf5"
          if [ ! -f "${resultdir}/${expected_file}" ]; then
              echo -e "NOT EXISTING: ${expected_file}\n*********** Submitting sbatch job ***********"
              echo "eval_glm_control_beta_native_space.sh $preproc $pcstop $frac $uid"
              sbatch eval_glm_control_beta_native_space.sh $preproc $pcstop $frac $uid
#          else
#              echo -e "EXISTING: $expected_file\n*********** Skipping sbatch job ***********"
          fi
      done
    done
  done
done