"""Figure out which unique parameter combinations were run"""

# Count unique GLMsingle parameters
pcstops_run1 = [1,2,3,4,5,6]
fracs_run1 = [0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0]
pcstops_run2 = [7,8,9,10]
fracs_run2 = [0.65, 0.55, 0.45, 0.35]
pcstops_run3 = [3,4,5,6,7,8,9,10]
fracs_run3 = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
pcstops_run4 = [3,4,5,6,7,8,9,10]
fracs_run4 = [0.05, 0.15]

# Combine pcstop and fracs into strings 'pcstop-{}_fracs-{}' for each run
pcstop_fracs_str_run1 = [f'pcstop-{pcstop}_fracs-{fracs}' for pcstop in pcstops_run1 for fracs in fracs_run1]
pcstop_fracs_str_run2 = [f'pcstop-{pcstop}_fracs-{fracs}' for pcstop in pcstops_run2 for fracs in fracs_run2]
pcstop_fracs_str_run3 = [f'pcstop-{pcstop}_fracs-{fracs}' for pcstop in pcstops_run3 for fracs in fracs_run3]
pcstop_fracs_str_run4 = [f'pcstop-{pcstop}_fracs-{fracs}' for pcstop in pcstops_run4 for fracs in fracs_run4]

# Get unique strings for all runs
pcstop_fracs_str_all = pcstop_fracs_str_run1 + pcstop_fracs_str_run2 + pcstop_fracs_str_run3 + pcstop_fracs_str_run4
pcstop_fracs_str_all = list(set(pcstop_fracs_str_all))

print(f'{len(pcstop_fracs_str_all)} unique parameter combinations: {pcstop_fracs_str_all}')