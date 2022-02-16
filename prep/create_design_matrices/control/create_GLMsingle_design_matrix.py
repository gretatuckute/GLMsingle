"""This script takes a data dict file and stimuli to create a design matrix for control type of experiment."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join
import pickle
import matplotlib.pyplot as plt

ROOT = '/Users/gt/Documents/GitHub/control-neural/'
DATADIR_DICT = (Path(ROOT) / 'data' / 'dict_neural_stimuli').resolve()
GLMDIR = '/Users/gt/Documents/GitHub/GLMsingle/'

EXPT = 'control'
PILOT = 'pilot3'
FL = 'control_tr1'

UID_TO_INCLUDE = ['853']
SESSION_TO_INCLUDE = ['FED_20211008a_3T1_PL2017', 'FED_20211013b_3T1_PL2017']
DATE_TAG_TO_INCLUDE = ['20220109']

# Load the data dict
with open(join(DATADIR_DICT, EXPT, PILOT, f"dict_UID-{'-'.join(UID_TO_INCLUDE)}_"
										  f"SESSION-{'-'.join(SESSION_TO_INCLUDE)}_FL-{FL}_{DATE_TAG_TO_INCLUDE[0]}.pkl"), 'rb') as f:
	data_dict = pickle.load(f)
	
# Load the magnitude responses based on the normalized ROIs
df_rois_normalized = data_dict['df_rois_normalized']
stimset = data_dict['stimset']

# Sort stimset according to what was presented when
stimset = stimset.sort_values(by=['session','run', 'trial_num' ])

n_trs = 168
n_runs = len(stimset.stim_set.unique())
n_items_in_run = len(stimset.trial_num.unique())
n_cond = n_runs * n_items_in_run

# Create n_runs lists of matrices size (n_trs, n_cond)
design_matrices = []
for run in range(n_runs):
	design_matrices.append(np.zeros((n_trs, n_cond)))
	
	
# Fill in the design matrices
for stim in stimset.itertuples():
	if stim.session == 1:
		run = 0 + stim.run - 1
	elif stim.session == 2:
		run = 10 + stim.run - 1
	else:
		raise ValueError('Stimulus session is not 1 or 2!')
	
	rel_time = int(stim.rel_time)
	if abs(stim.rel_time - int(stim.rel_time)) > 0.05:
		raise ValueError('Stimulus time is not close to an integer number -- timing might be off!')
	if (rel_time % 2 != 0): # check if rel_time is odd, which means there is something off
		raise ValueError('Relative time is not an even number!')
	
	tr_time = rel_time // 2
	
	design_matrices[run][tr_time, stim.item_id - 1] = 1 # just insert one in the correct position. The item id is the cond number.
	
# Check the sum of the design matrices. We expect the sum to be n_items_in_run*n_runs
assert (np.sum(design_matrices) == n_items_in_run*n_runs)

# Plot example design matrix
plt.figure(figsize=(20,20))
plt.imshow(design_matrices[0],interpolation='none')
plt.title('example design matrix from run 1',fontsize=18)
plt.xlabel('conditions',fontsize=18)
plt.ylabel('time (TR)',fontsize=18)
plt.tight_layout()
plt.show()

# Save design matrices and stimset_response_sorted
with open(join(GLMDIR, 'design_matrices', f"design_matrices_UID-{'-'.join(UID_TO_INCLUDE)}_"
						f"SESSION-{'-'.join(SESSION_TO_INCLUDE)}_FL-{FL}_{DATE_TAG_TO_INCLUDE[0]}_singletrial.pkl"), 'wb') as f:
	pickle.dump(design_matrices, f)

stimset.to_csv(join(GLMDIR, 'design_matrices', 'associated_stimsets', f"stimset_UID-{'-'.join(UID_TO_INCLUDE)}_"
						f"SESSION-{'-'.join(SESSION_TO_INCLUDE)}_FL-{FL}_{DATE_TAG_TO_INCLUDE[0]}_singletrial.csv"))
stimset.to_pickle(join(GLMDIR, 'design_matrices', 'associated_stimsets', f"stimset_UID-{'-'.join(UID_TO_INCLUDE)}_"
						f"SESSION-{'-'.join(SESSION_TO_INCLUDE)}_FL-{FL}_{DATE_TAG_TO_INCLUDE[0]}_singletrial.pkl"))
	





