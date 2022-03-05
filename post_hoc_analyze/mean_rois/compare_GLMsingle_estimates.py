"""Sript for comparing the estimates from GLMsingle one settings versus another."
Possibility of including traditional SPM GLM as well.
"""
import os

from utils import *

compare = 'all'
include_SPM = True
UID = '18'
norm = None

rois = ['lang_LH_netw', 'lang_RH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp',
		'aud_LH_netw']

### DIRECTORIES ###
GLMDIR = '/Users/gt/om/beta-neural-control/data/dict_neural_stimuli/' #'/Users/gt/Documents/GitHub/beta-neural-control/data/dict_neural_stimuli/'
SPMDIR = '/Users/gt/Documents/GitHub/pereira_modeling/'
SPM_fname = 'Pereira_FirstSession_TrialEffectSizes_20220223.csv'


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
	for roi in rois:
	
		save_str = f'betas-corr_{UID}_roi-{roi}_norm-{norm}'
		
		
		# Load all files in GLMDIR
		gs_files = [f for f in os.listdir(GLMDIR) if f.endswith('.pkl')]
		df_responses, gs_sents, lst_str_names = load_all_dicts(gs_files=gs_files,
															   GLMDIR=GLMDIR,
															   roi=roi,
															   norm=norm,)
		
		if include_SPM:
			spm_df, spm_sents = clean_up_SPM_estimates(SPM_fname,
													   SPMDIR=SPMDIR,
													   UID=UID,)
			assert (np.all(gs_sents == spm_sents))
			# Add SPM estimates to df_responses as new row
			# Give SPM estimates a new name
			df_responses = df_responses.append(spm_df, ignore_index=True)
			lst_str_names.append('SPM')
			# Rename last row
			# df_responses.iloc[-1].rename('beta_SPM')
	
	
		# Compute the correlation
		df_corr = df_responses.T.corr()
		df_corr.columns = lst_str_names
		df_corr.index = lst_str_names
		
		heatmap(df_corr,
				title=f'ROI: {roi}. Correlation of {roi} betas ({df_responses.shape[1]} items)\n'
					  f'Normalized by {norm}. UID: {UID}',
				save_str=save_str,
				save=True,
				vmin=0, vmax=1,)


