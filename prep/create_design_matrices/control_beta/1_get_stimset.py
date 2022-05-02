"""
Step 1: Obtain the stimulus set for the control beta experiment.
		Check and assert between the intended stimulus set (from fMRI_ready_materials) and the actual MATLAB output from the experiment.
		(Not done for UID 853, as this one is already checked and structured differently.)

"""

from utils_design_matrix import *

######## SETTINGS #########
FL = 'gs'
UID_TO_INCLUDE = 853
SESSION_TO_INCLUDE = d_UID_to_session_list[UID_TO_INCLUDE]

save = True

if save: # define save strings
	session_str = '-'.join(SESSION_TO_INCLUDE)
	save_str = f"UID-{UID_TO_INCLUDE}_SESSION-{session_str}_FL-{FL}_singletrial" # same save str for stimset and design matrices


######## CHECK, ADD ADDITIONAL INFO AND SAVE STIMSET (against actual experimental output) #########
if UID_TO_INCLUDE == 853:  # slightly different structure, load from two other sources (to check that they correspond)
	FL_name = 'control_tr1'
	MANUALDIR1 = f'{MRIPYTHONDIR}/prep_analyses/3_event2contrast'  # location 1, from mri_python
	MANUALDIR2 = f'{CONTROLNEURALDIR}/data/dict_neural_stimuli'  # location 2, from concatenated control_neural stimset
	
	# Load from MANUALDIR1:
	df_stim_manual1 = pd.read_csv(join(MANUALDIR1, f'itemid2contrast_{UID_TO_INCLUDE}_'
												   f'{SESSION_TO_INCLUDE[0]}_PL2017_'
												   f'{FL_name}.csv'))
	df_stim_manual1 = df_stim_manual1.sort_values(by=['session', 'run', 'trial_num'])
	
	df_stim_manual2 = pd.read_csv(join(MANUALDIR1, f'itemid2contrast_{UID_TO_INCLUDE}_'
												   f'{SESSION_TO_INCLUDE[1]}_PL2017_'
												   f'{FL_name}.csv'))
	df_stim_manual2 = df_stim_manual2.sort_values(by=['session', 'run', 'trial_num'])
	
	stimset1 = pd.concat([df_stim_manual1, df_stim_manual2])
	
	# Load the data dict
	with open(join(MANUALDIR2, 'control', 'pilot3', f"dict_UID-{UID_TO_INCLUDE}_"
													f"SESSION-{SESSION_TO_INCLUDE[0]}_PL2017-{SESSION_TO_INCLUDE[1]}_PL2017_"
													f"FL-{FL_name}_20220109.pkl"), 'rb') as f:
		data_dict = pickle.load(f)
	
	stimset2 = data_dict['stimset']
	# Sort stimset according to what was presented when
	stimset2 = stimset2.sort_values(by=['session', 'run', 'trial_num'])
	
	# Assert that stimset1 and stimset2 are the same
	assert (stimset1.item_id.values == stimset2.item_id.values).all()
	assert (stimset1.sentence.values == stimset2.sentence.values).all()
	assert (stimset1.trial_num.values == stimset2.trial_num.values).all()
	
	df_stimset = stimset2
	df_stimset = df_stimset.rename(columns={'session': 'session_id'})

else: # Other, recent UIDs (post 853 in 2021)
	
	# Load stimset from fMRI_ready_stimsets folder
	df_stimset = pd.read_csv(join(STIMSETDIR, f'{UID_TO_INCLUDE}', f'beta-control-neural_stimset_T_all_{UID_TO_INCLUDE}.csv'))

	# Load stimset from the actual experiment output to assert that the correct stimset per run was used
	runs_to_include = [f'{i:02d}' for i in range(1, n_runs+1)] # Add leading zero
	
	run_timestamps = [] # store the last timestamp of each run, and make sure that they were run consecutively
	lst_df_trials = []
	
	for i, run in enumerate(runs_to_include):
		if i < 10:
			session = SESSION_TO_INCLUDE[0]
		else:
			session = SESSION_TO_INCLUDE[1]
		
		# Load onsets table
		df_onsets = pd.read_csv(
			join(OUTPUTSDIR, f'control_UID-{UID_TO_INCLUDE}_session_id-{session}_run-{run}_data_onsets.csv'))
		
		df_trial = df_onsets.query('trial_type == "trial"')
		lst_df_trials.append(df_trial)
		
		# Assert that the onsets table is as expected
		run_timestamp = check_onsets_table(df_onsets=df_onsets,
						   run=run,
						   n_unique_trials_per_run=n_unique_trials_per_run,
						   fix_time=fix_time,
						   trial_time=trial_time,
						   break_index=break_index)
		
		run_timestamps.append(run_timestamp)

	# Assert that runs were run in consecutive order
	assert(run_timestamps == np.sort(run_timestamps)).all()

	# Merge the df trial tables
	df_trials = pd.concat(lst_df_trials)
	
	# Assert that it matches the other (full) stimset
	assert(df_trials.shape[0] == df_stimset.shape[0])
	assert(df_trials.item_id.values == df_stimset.item_id.values).all()
	assert(df_trials.sentence.values == df_stimset.sentence.values).all()
	
	# Examine which columns are not shared among the two dataframes
	df_stimset_cols = df_stimset.columns
	df_trials_cols = df_trials.columns
	
	df_trials_cols_not_in_stimset = df_trials_cols.difference(df_stimset_cols).values
	
	for col in df_trials_cols_not_in_stimset:
		if col == 'trial_type':
			continue
		
		df_stimset[col] = df_trials[col].values
		print(f'Added {col} to stimset')
		
	df_stimset = df_stimset.drop(columns=['index'])

### CREATE THE CORRECT INDEX TO THE STIMSET AND SESSION INDICATOR ###
df_stimset['uid'] = UID_TO_INCLUDE
df_stimset['durationtr'] = trial_time // 2
df_stimset['sessionindicator'] = df_stimset['session_id'] # Create session indicator (1 if first session, 2 if second session). Already exists as session_id
assert(df_stimset['sessionindicator'].value_counts().values[0] == n_unique_trials // 2) # assert that half of trials are session 1, other half session 2
df_stimset['session'] = [SESSION_TO_INCLUDE[0] if x.session_id == 1 else SESSION_TO_INCLUDE[1] for x in df_stimset.itertuples()]

# Create an index with the UID_SESSION-SESSION and then the item_id
stimset_index = [f'{x.uid}_{session_str}.{x.item_id}' for x in df_stimset.itertuples()]
df_stimset.index = stimset_index

if save:
	df_stimset.to_csv(f'stimset_{save_str}.csv')