import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import typing
from os.path import join
import matplotlib
matplotlib.rcParams['svg.fonttype'] = 'none'

PLOTDIR = os.path.abspath(join(os.path.dirname( __file__ ), '..', 'plots'))
CSVDIR = os.path.abspath(join(os.path.dirname( __file__ ), '..', 'csvs'))

def clean_up_SPM_estimates(SPM_fname: str = None,
						   SPMDIR: str = None,
						   UID: str = None,
						   roi: str = 'lang_LH_netw'):
	"""Load and clean up SPM estimates (column names according to Ben)."""
	
	spm_df = pd.read_csv(SPMDIR + SPM_fname)
	# Lower-case first letter of column names in SPM dataframe if starting with 'Lang'
	spm_df.columns = [c[0].lower() + c[1:] if c.startswith('Lang') else c for c in spm_df.columns]
	spm_df.columns = [c[:2].lower() + c[2:] if c.startswith('MD') else c for c in spm_df.columns]
	spm_df = spm_df.rename(columns={'lang_LH_Netw': 'lang_LH_netw',
									'lang_RH_Netw': 'lang_RH_netw',
									'md_LH_Netw': 'md_LH_netw',
									'md_RH_Netw': 'md_RH_netw'})
	
	# Format UID col as strings
	spm_df['UID'] = spm_df['UID'].astype(str)
	
	spm_vals = spm_df.query('UID == @UID')[roi]
	spm_sents = spm_df.query('UID == @UID')['Sentence'].str.replace('"', '').values
	
	return spm_vals, spm_sents

def load_all_dicts(gs_files: list = None,
				   GLMDIR: str = None,
				   norm: typing.Union[str, bool] = 'bySessVoxZ',
				   roi: str = 'lang_LH_netw'):
	
	# Load, extract the ROI of interest
	lst_responses = []
	lst_sents = []
	lst_str_names = []
	for f in np.sort(gs_files):
		# Obtain names of the params that change
		str_name = f.split('_')[-2:]
		modeltype = f.split('_')[5]
		str_name[-1] = str_name[-1].split('.pkl')[0]
		str_name.append(modeltype)
		str_name = '_'.join(str_name)
		lst_str_names.append(str_name)
		
		glm_d = pd.read_pickle(GLMDIR + f)
		glm_df = glm_d[f'df_rois_norm-{norm}']
		glm_vals = glm_df[roi].values
		lst_responses.append(glm_vals)
		
		glm_sents = glm_d['stimset']['sentence'].values
		lst_sents.append(glm_sents)
	
	df_responses = pd.DataFrame(lst_responses, index=lst_str_names)
	
	# Assert that all sents were identical (for each file)
	for i in range(len(lst_sents)):
		assert (np.all(lst_sents[i] == lst_sents[0]))
		
	return df_responses, lst_sents[0], lst_str_names

def heatmap(df_corr: pd.DataFrame = None,
			title: str = None,
			save_str: str = None,
			save: bool = False,
			vmin: float = None,
			vmax: float = None,):
	
	fig, ax = plt.subplots(figsize=(20, 20))
	sns.heatmap(df_corr,
				annot=False,
				fmt='.2f',
				ax=ax,
				cmap='RdBu_r',
				square=True,
				vmin=vmin,
				vmax=vmax,
				cbar_kws={'label': 'Pearson R',
						  'shrink': 0.5,
						  })
	plt.title(title)
	plt.tight_layout(pad=1)
	if save:
		plt.savefig(join(PLOTDIR, save_str + '.svg'), dpi=180)
		plt.savefig(join(PLOTDIR, save_str + '.png'), dpi=180)
		df_corr.to_csv(join(CSVDIR, save_str + '.csv'))
	plt.show()