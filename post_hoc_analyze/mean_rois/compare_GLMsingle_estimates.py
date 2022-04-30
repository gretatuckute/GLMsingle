"""Sript for comparing the estimates from GLMsingle one settings versus another."
Possibility of including traditional SPM GLM as well.
"""
from utils import *

compare = 'uidvsuid'
include_SPM = False
include_gs = True
UIDs = ['18', '288', '289', '296', '426']
norm = 'bySessVoxZ'

rois = ['lang_LH_netw', 'lang_RH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp',
		'aud_LH_netw']

### DIRECTORIES ###
GLMDIR = '/Users/gt/om/beta-neural-control/data/dict_neural_stimuli/' #'/Users/gt/Documents/GitHub/beta-neural-control/data/dict_neural_stimuli/'
SPMDIR = '/Users/gt/Documents/GitHub/pereira_modeling/'
SPM_fname = 'Pereira_FirstSession_TrialEffectSizes_20220223_fixed-col-names_wstimid.csv'


### 1 vs 1: COMPARE BETAS ###
if compare == '1vs1':
	pcstop1 = 5
	fracs1 = 0.5
	
	pcstop2 = 9
	fracs2 = 0.9
	
	GLM_fname1 = f'dict_{UID}_1-2_gs_thresh-90_type-d_preproc-swr_pcstop-{pcstop1}_fracs-{fracs1}.pkl'
	GLM_fname2 = f'dict_{UID}_1-2_gs_thresh-90_type-d_preproc-swr_pcstop-{pcstop2}_fracs-{fracs2}.pkl'
	
	### LOAD AND CLEAN UP DATA ###
	glm_d1 = pd.read_pickle(GLMDIR + GLM_fname1)
	glm_df1 = glm_d1[f'df_rois_norm-{norm}']
	glm_sents1 = glm_d1['stimset']['sentence'].values
	
	glm_d2 = pd.read_pickle(GLMDIR + GLM_fname2)
	glm_df2 = glm_d2[f'df_rois_norm-{norm}']
	glm_sents2 = glm_d2['stimset']['sentence'].values
	
	assert(np.all(glm_sents1 == glm_sents2))
	
	for roi in rois:
		glm_vals1 = glm_df1[roi].values
		glm_vals2 = glm_df2[roi].values
	
		print(f'ROI: {roi}: {np.corrcoef(glm_vals1, glm_vals2)[0][1]:.2f}')
	
	
### COMPARE OUTPUTS FROM SEVERAL BETAS, ONE ROI AT A TIME ###
if compare == 'all':
	for UID in UIDs:
		for roi in rois:
		
			save_str = f'betas-corr_{UID}_roi-{roi}_norm-{norm}'
			
			
			# Load all files in GLMDIR
			gs_files = [f for f in os.listdir(GLMDIR) if f.endswith('.pkl') and f.startswith(f'dict_{UID}')]
			
			# Only load modeltype d
			gs_files = [f for f in gs_files if 'type-d' in f]
			
			
			df_responses, gs_sents = load_all_dicts(gs_files=gs_files,
																   GLMDIR=GLMDIR,
																   roi=roi,
																   norm=norm,)
			
			
			if include_SPM:
				spm_df, spm_sents = get_SPM_estimates(SPM_fname,
													SPMDIR=SPMDIR,
													UID=UID,)
				assert (np.all(gs_sents == spm_sents))
				# Add SPM estimates to df_responses as new col
				df_responses['SPM'] = spm_df.values
	
		
		
			# Compute the correlation
			df_corr = df_responses.corr()
			
			heatmap(df_corr,
					title=f'ROI: {roi}. Correlation of {roi} betas ({df_responses.shape[1]} items)\n'
						  f'Normalized by {norm}. UID: {UID}',
					save_str=save_str,
					save=True,
					vmin=0, vmax=1,)

### COMPARE OUTPUT FROM ONE BETA INSTANTIATION, SUBJECT VS SUBJECT ###
if compare == 'uidvsuid':
	roi = 'lang_LH_netw'
	UID_str = '-'.join(UIDs)
	save_str = f'corr_{roi}_{UID_str}'
	
	if include_SPM:
		save_str += '_SPM'
		lst_spm_dfs = []
		lst_spm_sents = []
		for UID in UIDs:
			spm_df, spm_sents = get_SPM_estimates(SPM_fname,
												  SPMDIR=SPMDIR,
												  UID=UID,
												  roi=roi)
			lst_spm_dfs.append(spm_df.reset_index(drop=True))
			lst_spm_sents.append(spm_sents)
			
		
		# Assert that sentences match up
		assert ([spm_sents[0] == spm_sents[i] for i in range(len(lst_spm_sents))])
		df_across_uids = pd.concat(lst_spm_dfs, axis=1)
		
		df_corr = df_across_uids.corr()
		df_corr.index = UIDs
		df_corr.columns = UIDs
		
		heatmap(df_corr,
				title=f'ROI: {roi}. Correlation across {len(spm_sents)} items\n'
					  f'UIDs: {UID_str}',
				save_str=save_str,
				save=True,
				vmin=0, vmax=0.3,
				figsize=(5,5))
		
	pcstop = 7
	fracs = 0.7
	save_str = f'corr_{roi}_{UID_str}_pcstop-{pcstop}_fracs-{fracs}_norm-{norm}'
	
	# Load gs data across UIDs
	lst_gs_dfs = []
	lst_gs_sents = []
	for UID in UIDs:
		GLM_fname = f'dict_{UID}_1-2_gs_thresh-90_type-d_preproc-swr_pcstop-{pcstop}_fracs-{fracs}.pkl'
		
		### LOAD AND CLEAN UP DATA ###
		glm_d = pd.read_pickle(GLMDIR + GLM_fname)
		gs_df = glm_d[f'df_rois_norm-{norm}']
		gs_sents = glm_d['stimset']['sentence'].values
		lst_gs_sents.append(gs_sents)
		lst_gs_dfs.append(gs_df[roi].reset_index(drop=True))
	
	assert ([lst_gs_sents[0] == lst_gs_sents[i] for i in range(len(lst_gs_sents))])
	df_across_uids = pd.concat(lst_gs_dfs, axis=1)
	
	df_corr = df_across_uids.corr()
	df_corr.index = UIDs
	df_corr.columns = UIDs
	
	heatmap(df_corr,
			title=f'ROI: {roi}. Correlation across {len(gs_sents)} items\n'
				  f'UIDs: {UID_str}\n'
				  f'Pcstop: {pcstop}, fracs: {fracs}',
			save_str=save_str,
			save=True,
			vmin=0, vmax=0.3,
			figsize=(5, 5))
	
	

	
	