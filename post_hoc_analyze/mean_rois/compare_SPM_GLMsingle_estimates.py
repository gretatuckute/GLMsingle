"""Sript for comparing the estimates from SPM traditional betas versus GLMsingle betas"""

import pandas as pd
import numpy as np
import os

UID = '18'
rois = ['lang_LH_netw', 'lang_RH_netw', 'lang_LH_IFGorb', 'lang_LH_IFG', 'lang_LH_MFG', 'lang_LH_AntTemp', 'lang_LH_PostTemp', 'lang_LH_AngG']
norm = 'bySessVoxZ'
### DIRECTORIES ###
SPMDIR = '/Users/gt/Documents/GitHub/pereira_modeling/'
SPM_fname = 'Pereira_FirstSession_TrialEffectSizes_20220223.csv'

GLMDIR = '/Users/gt/Documents/GitHub/beta-neural-control/data/dict_neural_stimuli/'
GLM_fname = f'dict_{UID}_1-2_gs_thresh-90_type-d_preproc-swr_pcstop-5_fracs-0.5.pkl'

### LOAD AND CLEAN UP DATA ###
spm_df = pd.read_csv(SPMDIR + SPM_fname)
glm_d = pd.read_pickle(GLMDIR + GLM_fname)
glm_df = glm_d[f'df_rois_norm-{norm}']
glm_sents = glm_d['stimset']['sentence'].values

# Lower-case first letter of column names in SPM dataframe if starting with 'Lang'
spm_df.columns = [c[0].lower() + c[1:] if c.startswith('Lang') else c for c in spm_df.columns]
spm_df.columns = [c[:2].lower() + c[2:] if c.startswith('MD') else c for c in spm_df.columns]
spm_df = spm_df.rename(columns={'lang_LH_Netw': 'lang_LH_netw',
								'lang_RH_Netw': 'lang_RH_netw',
								'md_LH_Netw': 'md_LH_netw',
								'md_RH_Netw': 'md_RH_netw'})

# Format UID col as strings
spm_df['UID'] = spm_df['UID'].astype(str)
spm_sents = spm_df.query('UID == @UID')['Sentence'].str.replace('"', '').values

assert(np.all(spm_sents == glm_sents))

### COMPARE BETAS ###
for roi in rois:
	spm_vals = spm_df.query('UID == @UID')[roi].values
	glm_vals = glm_df[roi].values
	
	print(f'ROI: {roi}: {np.corrcoef(spm_vals, glm_vals)[0][1]:.2f}')


