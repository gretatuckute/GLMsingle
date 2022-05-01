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

def get_SPM_estimates(SPM_fname: str = None,
						   SPMDIR: str = None,
						   UID: str = None,
						   roi: str = 'lang_LH_netw'):
	"""Load and clean up SPM estimates (column names according to Ben)."""
	
	spm_df = pd.read_csv(SPMDIR + SPM_fname)
	spm_df['UID'] = spm_df['UID'].astype(str)
	spm_df = spm_df.query('UID == @UID').set_index('stim_id')
	
	# Reorder so itemid is ascending
	spm_df = spm_df.sort_values(by='itemid')
	index_itemid = [int(x.split('.')[-1]) for x in spm_df.index]
	assert (index_itemid == spm_df.itemid.values).all()
	spm_vals = spm_df[roi]
	spm_sents = spm_df['sentence'].values

	return spm_vals, spm_sents

def load_all_dicts(gs_files: list = None,
				   GLMDIR: str = None,
				   norm: typing.Union[str, bool] = 'bySessVoxZ',
				   roi: str = 'lang_LH_netw'):
	"""Return df of rows = sentence, cols = gs single response for ROI of interest"""
	
	# Load, extract the ROI of interest
	lst_responses = []
	lst_sents = []
	lst_str_names = []
	
	# Make sure that the files are sorted as 0, 1, 2, ... and not 0 to 10
	# Pad pcstop- with 0s
	
	for f in np.sort(gs_files):
		# Obtain names of the params that change
		str_name = f.split('_')[-2:]
		if len(str_name[0]) < 9:
			str_name[0] = str_name[0].split('-')[0] + '-0' + str_name[0].split('-')[-1]
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
	df_responses = df_responses.sort_index()
	
	# Assert that all sents were identical (for each file)
	for i in range(len(lst_sents)):
		assert (np.all(lst_sents[i] == lst_sents[0]))
		
	return df_responses.T, lst_sents[0]

def heatmap(df_corr: pd.DataFrame = None,
			title: str = None,
			save_str: str = None,
			save: bool = False,
			vmin: float = None,
			vmax: float = None,
			figsize: tuple = (20, 20),):
	
	fig, ax = plt.subplots(figsize=figsize)
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