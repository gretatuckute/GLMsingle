import os
import pandas as pd
import numpy as np
from os.path import join
import pickle
import matplotlib.pyplot as plt
from getpass import getuser

#### RESOURCES ####
user = getuser()

if user == 'gt':
	GLMDIR = '/Users/gt/Documents/GitHub/GLMsingle/'
	STIMSETDIR = '/Users/gt/Documents/GitHub/beta-neural-control/material_selection/fMRI_ready_stimsets/' # where stimsets are stored
	OUTPUTSDIR = '/Volumes/GoogleDrive/My Drive/Research2020/control/EXPERIMENT_RELATED/beta-neural-control/outputs/' # where outputs from MATLAB expt are stored, locally (from Franklin)
	MRIPYTHONDIR = '/Users/gt/Documents/GitHub/mri_python/'
	CONTROLNEURALDIR = '/Users/gt/Documents/GitHub/control-neural/'
elif user == 'gretatu':
	GLMDIR = '/om5/group/evlab/u/gretatu/GLMsingle/'
	STIMSETDIR = '/om/user/gretatu/beta-neural-control/material_selection/fMRI_ready_stimsets/'  # where stimsets are stored
	OUTPUTSDIR = '/Volumes/GoogleDrive/My Drive/Research2020/control/EXPERIMENT_RELATED/beta-neural-control/outputs/'  # where outputs from MATLAB expt are stored, locally (from Franklin)
	MRIPYTHONDIR = '/om5/group/evlab/u/gretatu/mri_python/'
	CONTROLNEURALDIR = '/om/user/gretatu/control-neural/'
else:
	raise Exception('User not recognized')


d_UID_to_session_list = {848: ['FED_20220420b_3T1', 'FED_20220427a_3T1'],
						 853: ['FED_20211008a_3T1', 'FED_20211013b_3T1'],
						 865: ['FED_20220414b_3T1', 'FED_20220415a_3T1'],
						 875: ['FED_20220408a_3T1', 'FED_20220411a_3T1'],
						 876: ['FED_20220413a_3T1', 'FED_20220420a_3T1']}

## Expected experiment params ##
n_trs = 168
n_runs = 20
n_unique_trials_per_run = 50
n_unique_trials = n_unique_trials_per_run * n_runs
fix_time = 4 # seconds
trial_time = 2 # seconds
break_index = 50


def check_onsets_table(df_onsets: pd.DataFrame = None,
					   run: str = '',
					   n_unique_trials_per_run: int = 50,
					   fix_time: int = 4,
					   trial_time: int = 2,
					   break_index: int = 50):
	
	df_trial = df_onsets.query('trial_type == "trial"')
	df_fix = df_onsets.query('trial_type == "fix"')
	
	assert df_onsets.shape[0] == n_unique_trials_per_run * 2 # trial and fix are stored as separate rows
	assert(df_trial.shape[0] == n_unique_trials_per_run)
	assert(df_fix.shape[0] == n_unique_trials_per_run)
	assert (len(df_trial.sentence.unique()) == n_unique_trials_per_run)
	
	# Assert that fixation is fix_time
	time_between_trials = df_trial['rel_time'].diff()
	
	# find largest difference between actual and expected time between trial onset
	diff_between_actual_and_expected_time = abs(((fix_time + trial_time) - time_between_trials))
	
	# Assert that there's just one nan, for the 1st difference and no other nans
	assert(np.isnan(diff_between_actual_and_expected_time[0]))
	assert(np.isnan(diff_between_actual_and_expected_time[1:]).sum() == 0)
	
	diff_between_actual_and_expected_time[0] = 0 # first trial is not preceded by a fixation, otherwise gives nan
	
	# There should be one difference that is 12s (midway fix) and then the rest should be zero
	assert(np.allclose(diff_between_actual_and_expected_time[break_index], 12, atol=1e-2))
	
	# Remove break index
	diff_between_actual_and_expected_time[break_index] = 0
	
	
	print \
		(f'Passed assertions for run {run}. Largest difference between actual and expected time between trials: {diff_between_actual_and_expected_time.max():.5f}')
	
	# Assert that rel_time and trial_num are ascending
	assert (np.all(df_trial['rel_time'].values == df_trial['rel_time'].sort_values()))
	assert (np.all(df_trial['trial_num'].values == df_trial['trial_num'].sort_values()))
	
	run_timestamp = df_trial['time'].max()
	
	return run_timestamp