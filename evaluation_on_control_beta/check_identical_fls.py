"""
Check how ridgefrac scaling affects the first level betas. Requires loading netw_dict data from GLMsingle.
"""
from utils_voxels import *

netw_dict_a = pd.read_pickle('/Users/gt/Desktop/output_glmsingle_preproc-swr_pcstop-3_fracs-0.25_UID-853/'
							 'netw_dict_853_FED_20211008a_3T1-FED_20211013b_3T1_gs_thresh-90_mask-langloc_type-d_preproc-swr_pcstop-3_fracs-0.25.pkl')
netw_dict_b = pd.read_pickle('/Users/gt/Desktop/output_glmsingle_preproc-swr_pcstop-3_fracs-0.95_UID-853/'
							 'netw_dict_853_FED_20211008a_3T1-FED_20211013b_3T1_gs_thresh-90_mask-langloc_type-d_preproc-swr_pcstop-3_fracs-0.95.pkl')
netw_dict_c = pd.read_pickle('/Users/gt/Desktop/output_glmsingle_preproc-swr_pcstop-10_fracs-0.55_UID-853/'
							 'netw_dict_853_FED_20211008a_3T1-FED_20211013b_3T1_gs_thresh-90_mask-langloc_type-d_preproc-swr_pcstop-10_fracs-0.55.pkl')

roi = 'lang_LH_AntTemp'

roi_vox_a = netw_dict_a['neural_data'].loc[:, netw_dict_a['neural_meta'].roi == roi]
roi_vox_b = netw_dict_b['neural_data'].loc[:, netw_dict_b['neural_meta'].roi == roi]
roi_vox_c = netw_dict_c['neural_data'].loc[:, netw_dict_c['neural_meta'].roi == roi]

# Plot just one voxel
vox_idx = 100
vox_a = roi_vox_a.iloc[:, vox_idx]
vox_b = roi_vox_b.iloc[:, vox_idx]
vox_c = roi_vox_c.iloc[:, vox_idx]

plt.figure()
plt.plot(vox_a, label='preproc-swr_pcstop-3_fracs-0.25', alpha=0.5)
plt.plot(vox_b, label='preproc-swr_pcstop-3_fracs-0.95', alpha=0.5)
# plt.plot(vox_c, label='preproc-swr_pcstop-10_fracs-0.55')
plt.legend()
plt.title(f'{roi}, one voxel, across all conditions')
plt.show()

# What is the mean of that voxel across all conditions?
vox_a_mean = vox_a.mean()
vox_b_mean = vox_b.mean()
vox_c_mean = vox_c.mean()


# Get means of the roi voxels and plot them
roi_vox_a_mean = roi_vox_a.mean(axis=1)
roi_vox_b_mean = roi_vox_b.mean(axis=1)
roi_vox_c_mean = roi_vox_c.mean(axis=1)

plt.figure(figsize=(10,5))
plt.plot(roi_vox_a_mean, label='pcstop=3, fracs=0.25', alpha=0.5)
plt.plot(roi_vox_b_mean, label='pcstop=3, fracs=0.95', alpha=0.5)
# plt.plot(roi_vox_c_mean, label='pcstop=10, fracs=0.55')
plt.legend()
plt.title(f'ROI: {roi}; UID 853')
plt.show()

# Mean using axis 0 (means across all items)
roi_vox_a_mean_0 = roi_vox_a.mean(axis=0)
roi_vox_b_mean_0 = roi_vox_b.mean(axis=0)
roi_vox_c_mean_0 = roi_vox_c.mean(axis=0)

# Check whether they are identical
print((np.abs(roi_vox_a_mean_0 - roi_vox_b_mean_0)).max())
print((np.abs(roi_vox_a_mean_0 - roi_vox_c_mean_0)).max())
print((np.abs(roi_vox_b_mean_0 - roi_vox_c_mean_0)).max())



plt.figure(figsize=(10,5))
plt.plot(roi_vox_a_mean_0, label='pcstop=3, fracs=0.25', alpha=0.5)
plt.plot(roi_vox_b_mean_0, label='pcstop=3, fracs=0.95', alpha=0.5)
plt.plot(roi_vox_c_mean_0, label='pcstop=10, fracs=0.55')
plt.legend()
plt.title(f'ROI: {roi}; UID 853, mean across all items')
plt.show()


# How correlated are the voxels in the two networks?
corr_ab = roi_vox_a.corrwith(roi_vox_b)
corr_ac = roi_vox_a.corrwith(roi_vox_c)
corr_bc = roi_vox_b.corrwith(roi_vox_c)

# Which correlation is the best?
print('corr_ab:', corr_ab.mean())
print('corr_ac:', corr_ac.mean())
print('corr_bc:', corr_bc.mean())


# Check new_nii files
nii_a = np.array(nib.load('/Users/gt/Desktop/output_glmsingle_preproc-swr_pcstop-3_fracs-0.25_UID-853/'
						  'betas-mean_853_FED_20211008a_3T1-FED_20211013b_3T1_gs_thresh-90_mask-langloc_type-d_preproc-swr_pcstop-6_fracs-0.25.nii').dataobj).flatten()
nii_b = np.array(nib.load('/Users/gt/Desktop/output_glmsingle_preproc-swr_pcstop-3_fracs-0.95_UID-853/'
						  'betas-mean_853_FED_20211008a_3T1-FED_20211013b_3T1_gs_thresh-90_mask-langloc_type-d_preproc-swr_pcstop-6_fracs-0.95.nii').dataobj).flatten()



# First obtain the neuroid (indices in the flattened 91, 109, 91 matrix) of the ROI
roi_indices = netw_dict_a['neural_meta'].index[netw_dict_a['neural_meta'].roi == roi].values

# Index into nii_a and nii_b to get the voxel values of the ROI
nii_a_roi = nii_a[roi_indices]
nii_b_roi = nii_b[roi_indices]

# Plot the voxel values of the ROI
plt.figure(figsize=(10,5))
plt.plot(nii_a_roi, label='pcstop=3, fracs=0.25', alpha=0.5)
plt.plot(nii_b_roi, label='pcstop=3, fracs=0.95', alpha=0.5)
plt.legend()
plt.title(f'ROI: {roi}; UID 853; betas-mean, new_nii')
plt.show()