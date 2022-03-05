"""This script takes a data dict file and stimuli to create a design matrix for Pereira type of experiment.
Also store associated stimset"""

import os
import pandas as pd
import numpy as np
from os.path import join
import pickle
import matplotlib.pyplot as plt

GLMDIR = '/Users/gt/Documents/GitHub/GLMsingle/'
FL = 'gs'

UID_TO_INCLUDE = 426

d_UID_to_session_list = {18: ['FED_20151130a_3T1', 'FED_20160112d_3T2'],
						 288: ['FED_20151106a_3T1', 'FED_20151201a_3T1'],
						 289: ['FED_20150908b_3T2', 'FED_20151207a_3T1'],
						 296: ['FED_20151030a_3T1', 'FED_20151130b_3T1'],
						 426: ['FED_20170126c_3T2', 'FED_20170307b_3T2']}

SESSION_TO_INCLUDE = d_UID_to_session_list[UID_TO_INCLUDE]

# Load the data dict
data_dict = pd.read_csv('Pereira_FirstSession_SingleTrialTiming_20220215_wIPS_witemid.csv')

# Get UID and sessions of interest
df = data_dict.query('UID == @UID_TO_INCLUDE and Session in @SESSION_TO_INCLUDE')

# Sort stimset according to what was presented when
stimset = df.sort_values(by=['Session','DicomNumber', 'OnsetTR'])
assert(len(stimset.Session.unique()) == len(SESSION_TO_INCLUDE))

# Create col that is a unique indexer for Session and DicomNumber
stimset['Session_DicomNumber'] = stimset.Session + '_' + stimset.DicomNumber.astype(str)

n_runs = len(stimset.Session_DicomNumber.unique())

# Make sure dicom images are in the right order
unique_runs = stimset.Session_DicomNumber.unique() # list of unique runs for this subject. Because they are sorted, the index should be retained, but let's check:
_, idx = np.unique(stimset.Session_DicomNumber, return_index=True)
unique_runs_check = stimset.Session_DicomNumber.values[np.sort(idx)]
assert(np.all(unique_runs == unique_runs_check))

n_cond = len(stimset.Stim.unique())
print(f'Number of runs: {n_runs}, number of conditions: {n_cond} for UID {UID_TO_INCLUDE} across {SESSION_TO_INCLUDE}\n')

# Create n_runs lists of matrices size (n_trs, n_cond)
design_matrices = []
for run in unique_runs:
	run_of_interest = stimset.query('Session_DicomNumber == @run')
	n_trs = run_of_interest.IPS.unique()
	assert(len(n_trs) == 1)
	n_trs = int(n_trs[0])
	
	# Fill in the design matrices
	design_matrix = np.zeros((n_trs, n_cond))
	for item in run_of_interest.itemid.values:
		stim_of_interest = run_of_interest.query('itemid == @item')
		onset = stim_of_interest.OnsetTR.values
		duration = stim_of_interest.DurationTR.values
		assert(len(duration) == 1)
		# just insert one in the correct position. The item id is the cond number.
		design_matrix[(onset)-1, item] = 1 # If duration = 2 TR, then we DON'T want to occupy the second TR (GLMsingle takes care of this if we specify stimdur as 4s (2TRs))
		# the subtraction by 1 is because of python indexing. If the TR onset is 5, then we want that to be 5 in python indexing too (and not index 6)
	design_matrices.append(design_matrix)
	
# Check the sum of the design matrices. We expect the sum to be n_items_in_run*n_runs
sum_design_matrices = int(np.sum([np.sum(x) for x in design_matrices], axis=0))
assert (sum_design_matrices == n_cond)

# Plot example design matrix
plt.figure(figsize=(20,20))
plt.imshow(design_matrices[0],interpolation='none')
plt.title('example design matrix from run 1',fontsize=18)
plt.xlabel('conditions',fontsize=18)
plt.ylabel('time (TR)',fontsize=18)
plt.tight_layout()
plt.show()

### Save design matrices ###
session_str = '-'.join(SESSION_TO_INCLUDE)
uid_str = '-'.join([str(x) for x in [UID_TO_INCLUDE]])
save_str = f"UID-{uid_str}_SESSION-{session_str}_FL-{FL}_singletrial" # same save str for stimset and design matrices

with open(join(GLMDIR, 'design_matrices', f'design_matrices_{save_str}.pkl'), 'wb') as f:
	pickle.dump(design_matrices, f)
	print(f'Saved design matrices to {join(GLMDIR, "design_matrices", f"design_matrices_{save_str}.pkl")}')

### Save stimset ###
stimset_save = stimset.copy(deep=True)
stimset_save.drop(columns=['Unnamed: 0'], inplace=True)
# Get rid of "" in the Sentence col
stimset_save.Sentence = stimset_save.Sentence.str.replace('"', '')
# Create run columns
stimset_save['run'] = stimset_save.Event.apply(lambda x: x.split('run')[-1].split('_')[0])

# Create an index with the UID_SESSION-SESSION and then the itemid
stimset_index = [f'{x.UID}_{session_str}.{x.itemid}' for x in stimset_save.itertuples()]
stimset_save.index = stimset_index

# Create session indicator (1 if first session, 2 if second session)
stimset_save['sessionindicator'] = np.where(stimset_save.Session == SESSION_TO_INCLUDE[0], 1, 2)

# Rename the columns, make all lowercase
stimset_save.columns = [x.lower() for x in stimset_save.columns]

stimset_save.to_csv(join(GLMDIR, 'design_matrices', 'associated_stimsets', f'stimset_{save_str}.csv'))
stimset_save.to_pickle(join(GLMDIR, 'design_matrices', 'associated_stimsets', f'stimset_{save_str}.pkl'))
print(f'Saved stimset to {join(GLMDIR, "design_matrices", "associated_stimsets", f"{save_str}.pkl")}')





