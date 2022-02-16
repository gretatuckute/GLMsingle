import pandas as pd
import numpy as np
from pathlib import Path
from os.path import join
import os

"""Load file with UID and session info and find IPS for dicom number of interest"""

ROOT = '/mindhive/evlab/u/Shared/SUBJECTS/'

# Load the data dict
data_dict = pd.read_csv('Pereira_FirstSession_SingleTrialTiming_20220215.csv')
data_dict['IPS'] = np.nan
data_dict['nii_wr_path'] = np.nan
data_dict['nii_swr_path'] = np.nan

# Find IPS for each unique UID and session and dicom number
for i, row in data_dict.iterrows():
	uid = str(row['UID'])
	if len(uid) < 3: # prefix 0
		uid = '0' + uid
	session = str(row['Session'])
	dicom = (row['DicomNumber'])
	print(uid, session, dicom)
	# Load the dicom summary file
	data_path = join(ROOT, f'{uid}_{session}_PL2017', 'dicom_summary.csv')
	data = pd.read_csv(data_path,index_col=False)
	# Find IPS for RUN_NUM of interest
	ips = int(data.loc[data['RUN_NUM'] == dicom, 'IPS'].values)
	# Also obtain the dicom path
	nii_path = join(ROOT, f'{uid}_{session}_PL2017', 'nii')
	# list all files in directory
	files = os.listdir(nii_path)
	files = [f for f in files if f.endswith('.nii')]
	wr_name = [f for f in files if f.startswith('wr') and f.endswith(f'-{dicom}.nii')][0]
	swr_name = [f for f in files if f.startswith('swr') and f.endswith(f'-{dicom}.nii')][0]
	print(wr_name)
	data_dict.loc[i, 'nii_wr_path'] = join(nii_path, wr_name)
	data_dict.loc[i, 'nii_swr_path'] = join(nii_path, swr_name)
	# print(f'\n\n{ips}')
	# Save the IPS
	data_dict.loc[i, 'IPS'] = int(ips)
	
# Save the data dict
data_dict.to_csv('Pereira_FirstSession_SingleTrialTiming_20220215_wIPS.csv')

