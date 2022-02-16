import os
import pandas as pd
from os.path import join
import pickle

# Load the data dict
data_dict = pd.read_csv('Pereira_FirstSession_SingleTrialTiming_20220215_wIPS.csv')

# Create an itemid that is unique for each Stim (integer)
# Create mapping that is unique for each Stim (string) to Int (integer)
stim_to_int = {}
for i, stim in enumerate(data_dict['Stim']):
	if stim not in stim_to_int:
		stim_to_int[stim] = i
# Save the mapping
with open(join('pereira_stim_to_itemid.pickle'), 'wb') as f:
	pickle.dump(stim_to_int, f)

# Now populate the stimset with the unique itemids
itemids = []
for i, stim in enumerate(data_dict['Stim']):
	itemids.append(stim_to_int[stim])

data_dict['itemid'] = itemids

# Save the data dict
data_dict.to_csv(join('Pereira_FirstSession_SingleTrialTiming_20220215_wIPS_witemid.csv'), index=False)