"""This script takes a data dict file and stimuli to create a design matrix control_beta.
Also stores associated stimset.

Performs assertions based on the output onsets_table.
"""

from utils_design_matrix import *

######## SETTINGS #########
GLMDIR = '/Users/gt/Documents/GitHub/GLMsingle/'
STIMSETDIR = '/Users/gt/Documents/GitHub/beta-neural-control/material_selection/fMRI_ready_stimsets/' # where stimsets are stored
OUTPUTSDIR = '/Volumes/GoogleDrive/My Drive/Research2020/control/EXPERIMENT_RELATED/beta-neural-control/outputs/' # where outputs from MATLAB expt are stored, locally (from Franklin)
FL = 'gs'

UID_TO_INCLUDE = 848
SESSION_TO_INCLUDE = d_UID_to_session_list[UID_TO_INCLUDE]

## Expected experiment params ##
n_trs = 168
n_runs = 20
n_unique_trials_per_run = 50
n_unique_trials = n_unique_trials_per_run * n_runs
fix_time = 4 # seconds
trial_time = 2 # seconds
break_index = 50

save = True
if save: # define save strings
	session_str = '-'.join(SESSION_TO_INCLUDE)
	save_str = f"UID-{UID_TO_INCLUDE}_SESSION-{session_str}_FL-{FL}_singletrial" # same save str for stimset and design matrices


######## CHECK, ADD ADDITIONAL INFO AND SAVE STIMSET (against actual experimental output) #########
if UID_TO_INCLUDE == 853:  # slightly different structure, load from two other sources (to check that they correspond)
	FL_name = 'control_tr1'
	MANUALDIR1 = '/Users/gt/Documents/GitHub/mri_python/prep_analyses/3_event2contrast'  # location 1, from mri_python
	MANUALDIR2 = '/Users/gt/Documents/GitHub/control-neural/data/dict_neural_stimuli'  # location 2, from concatenated control_neural stimset
	
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

### ADD THE CORRECT INDEX TO THE STIMSET, FILE PATHS, SESSION INDICATOR ###

# Create an index with the UID_SESSION-SESSION and then the item_id
stimset_index = [f'{x.UID}_{session_str}.{x.item_id}' for x in df_stimset.itertuples()]
df_stimset.index = stimset_index

# Create session indicator (1 if first session, 2 if second session)
df_stimset['sessionindicator'] = np.where(df_stimset.session_id == SESSION_TO_INCLUDE[0], 1, 2)

# Rename the columns, make all lowercase
df_stimset.columns = [x.lower() for x in df_stimset.columns]







if save: # Store the stimset with additional columns in the associated_stimset dir
	df_stimset.to_csv(join(GLMDIR, 'design_matrices', 'associated_stimsets', f'stimset_{save_str}.csv'), index=False)
	df_stimset.to_pickle(join(GLMDIR, 'design_matrices', 'associated_stimsets', f'stimset_{save_str}.pkl'), protocol=4)
	print(f'Saved stimset to design_matrices/associated_stimsets/stimset_{save_str}.csv')
	
######## CREATE DESIGN MATRIX BASED ON STIMSET ##########

# Create n_runs lists of matrices size (n_trs, n_cond)
design_matrices = []
for run in range(n_runs):
	design_matrices.append(np.zeros((n_trs, n_unique_trials)))

# Fill in the design matrices
for stim in df_stimset.itertuples():
	run_idx = stim.run_id - 1 # python indexed
	rel_time = int(stim.rel_time)
	if abs(stim.rel_time - int(stim.rel_time)) > 0.05:
		raise ValueError('Stimulus time is not close to an integer number -- timing might be off!')
	if (rel_time % 2 != 0):  # check if rel_time is odd, which means there is something off
		raise ValueError('Relative time is not an even number!')
	
	tr_time = rel_time // 2
	
	design_matrices[run_idx][
		tr_time, stim.item_id - 1] = 1  # just insert one in the correct position
	# For item_ids: So we have 1,000 conditions in total, and given python indexing, we have to fill in from 0 to 999.
	# The itemIDs are not zero-indexed.
	# For tr_time: First, we have 12s (6TR) of fixation. tr_time: 12 // 2 = 6. So, filling in the trial at TR 6 is correct,
	# given that the first trial actually takes place at TR 7. So, filling in at 6 yields the 7th TR.

# Check the sum of the design matrices. We expect the sum to be n_items_in_run*n_runs
assert (np.sum(design_matrices) == n_unique_trials_per_run * n_runs)

# Plot example design matrix
plt.figure(figsize=(20, 20))
plt.imshow(design_matrices[0], interpolation='none')
plt.title('example design matrix from run 1', fontsize=18)
plt.xlabel('conditions', fontsize=18)
plt.ylabel('time (TR)', fontsize=18)
plt.tight_layout()
plt.show()

if save:
	# Save design matrices
	with open(join(GLMDIR, 'design_matrices', f"design_matrices_{save_str}.pkl"), 'wb') as f:
		pickle.dump(design_matrices, f, protocol=4)
	
	print(f'Saved design matrices to {join(GLMDIR, "design_matrices")} as design_matrices_{save_str}.pkl')
	
	
	