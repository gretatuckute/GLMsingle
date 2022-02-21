# # Find optimal GLMsingle parameters based on Pereira et al. (2018)
# ##### GLMsingle is new tool that provides efficient, scalable, and accurate single-trial fMRI response estimates.
# 
# Input: swr* files OR wr* files
# Parameters to test: HRFs, different number of PCs, fracridge parameter.
# 
# Every single trial is its own condition (i.e., 627 unique conditions), which is the item_id. So, if the first sentence presented in session 1, run 1, was item_id 994, then that is the condition.
# 

# ### Import function libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

import os
from os.path import join, exists, split
import sys
import time
import copy
import warnings
from tqdm import tqdm
from pprint import pprint
import sys
import datetime
import getpass
import argparse

warnings.filterwarnings('ignore')


def main(raw_args=None):
    # Mapping specific
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--UID', default='18', type=str, help='UID str')
    parser.add_argument('--session_str', default='FED_20151130a_3T1-FED_20160112d_3T2', type=str, help='Session str. Without PL2017')
    parser.add_argument('--FL', default='tr2', type=str, help='TR str')
    parser.add_argument('--datetag', default='20220215', type=str, help='Datetag str')
    parser.add_argument('--stimdur', default=4, type=int, help='Stimulus duration in seconds')
    parser.add_argument('--tr', default=2, type=int, help='TR sampling rate')
    parser.add_argument('--preproc', default='swr', type=str,
                        help='Which preprocessing pipeline to use. Default is swr.')
    parser.add_argument('--pcstop', default=1, type=int,
                        help='How many PCs to remove')
    parser.add_argument('--fracs', default=0.4, type=float,
                        help='Fraction of ridge regularization to use')
    parser.add_argument('--test', default=False, type=bool,
                        help='Whether to run test mode and only use one run for testing')
    parser.add_argument('--verbose', default=False, type=bool,
                        help='Whether to print output and not create a log file')
    parser.add_argument('--overwrite', default=True, type=bool,
                        help='Whether to overwrite results in case outputdir already exists')
    args = parser.parse_args(raw_args)
    
    # ### Set paths and download the example dataset
    user = getpass.getuser()
    print(f'Running as user {user}')
    if user != 'gt':
        root = '/om5/group/evlab/u/gretatu/GLMsingle/'
    else:
        root = '/Users/gt/om5/GLMsingle/'
    os.chdir(join(root))

    import glmsingle
    from glmsingle.glmsingle import GLM_single
    plot = False

    # arguments to change in GLM
    pcstop = -args.pcstop
    fracs = args.fracs

    # data args
    preproc = args.preproc
    
    # get metadata about stimulus duration and TR
    
    # create directory for saving data
    outputdir = join(root, 'output_glmsingle', f'output_glmsingle_preproc-{preproc}_pcstop{pcstop}_fracs-{fracs}_UID-{args.UID}')
    designdir = join(root, 'design_matrices')   # set design matrix directory
    stimsetdir = join(designdir, 'associated_stimsets')    # set stimset directory
    logdir = join(root, 'logs')

    if user != 'gt' and not args.verbose:
        date = datetime.datetime.now().strftime("%Y%m%d-%T")
        sys.stdout = open(join(logdir, f'out_{preproc}_pcstop{pcstop}_fracs-{fracs}_UID-{args.UID}_{date}.log'), 'a+')
    
    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    print(f'Preprocessing pipeline: {preproc} with {pcstop} PCs and {fracs} fracridge')
    print(f'\nSave output dir: {outputdir}')
    print(f'\nDesign matrices dir: {designdir}')
    print(f'\nLog dir: {logdir}\n')

    ### Organize BOLD data, design matrices, metadata
    SESSIONS = args.session_str.split('-')
    if not SESSIONS[0].endswith('PL2017'):
        SESSIONS = [x + '_PL2017' for x in SESSIONS]
    if args.test:
        print(f'Running in test mode. Only using one session: {SESSIONS[:1]}')
        SESSIONS = SESSIONS[:1]

    # Design matrix
    load_str = f'UID-{args.UID}_SESSION-{args.session_str}_FL-{args.FL}_{args.datetag}_singletrial'
    design = pd.read_pickle(join(designdir,  f'design_matrices_{load_str}.pkl'))
    
    # Associated stimset
    stimset = pd.read_pickle(join(stimsetdir, f'stimset_{load_str}.pkl'))
    # want to ensure that the loaded neural data matches the stimset and design matrix
    
    data = []
    session_indicators = []
    # Load each unique dicom image
    images_of_interest = stimset[f'nii_{args.preproc}_path'].unique()

    for i, s in enumerate(images_of_interest):
        print(f'Loading data from session UID: {args.UID}, dicom image: {s}\n')
        if args.test:
            if i == 7:
                break

        file = np.array(nib.load(s).dataobj)
        assert (file.shape[3] == design[i].shape[0])
        data.append(file)
        session_indicator = stimset.loc[stimset[f'nii_{args.preproc}_path'] == s, 'sessionindicator'].values[0]
        session_indicators.append(session_indicator)
        
    # get shape of data volume (XYZ) for convenience
    xyz = data[0].shape[:3]
    xyzt = data[0].shape
    
    print(f'Number of runs in data: {len(data)}.\nShape of Images (brain XYZ and TR): {data[0].shape}')
    if args.test:
        design = design[:7]
    
    print(f'Number of runs in design matrix: {len(design)}, with unique number of TRs across runs: {np.unique([x.shape[0] for x in design])}\n'
          f'and unique number of conditions: {np.unique([x.shape[1] for x in design])}\n'
          f'TR: {args.tr} and stimulus duration (in seconds): {args.stimdur}')

    assert (len(data) == len(design))
    assert (xyzt[-1] == design[0].shape[0])
    sys.stdout.flush()

    # ### Visualize sample data and design matrix

    if plot:
        # plot example slice from run 1
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.imshow(data[0][:, :, 50, 0])
        plt.title('example slice from run 1', fontsize=16)
        plt.subplot(122)
        plt.imshow(data[11][:, :, 50, 0])
        plt.title('example slice from run 12', fontsize=16)
    
        # plot example design matrix from run 1
        plt.figure(figsize=(20, 20))
        plt.imshow(design[0], interpolation='none')
        plt.title('example design matrix from run 1', fontsize=16)
        plt.xlabel('conditions', fontsize=16)
        plt.ylabel('time (TR)', fontsize=16);

    # print some relevant metadata
    print(f'There are {len(data)} runs in total\n')
    print(f'N = {data[0].shape[3]} TRs per run\n')
    print(f'The dimensions of the data for each run are: {data[0].shape}\n')
    print(f'The stimulus duration is {args.stimdur} seconds (TR={args.tr})\n')
    print(f'XYZ dimensionality is: {data[0].shape[:3]}\n')
    print(f'Numeric precision of data is: {type(data[0][0, 0, 0, 0])}\n')
    # print(f'There are {np.sum(roi)} voxels in the included visual ROI')

    # ### Run GLMsingle with default parameters to estimate single-trial betas

    # create a directory for saving GLMsingle outputs
    opt = dict()

    # set important fields for completeness (but these would be enabled by default)
    opt['wantlibrary'] = 1
    opt['wantglmdenoise'] = 1
    opt['wantfracridge'] = 1

    # for the purpose of this example we will keep the relevant outputs in memory
    # and also save them to the disk
    opt['wantfileoutputs'] = [1, 1, 1, 1]
    opt['wantmemoryoutputs'] = [1, 1, 1, 1]

    # add sessionindicator
    opt['sessionindicator'] = session_indicators

    # add changing parameters
    opt['pcstop'] = pcstop
    opt['fracs'] = fracs

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    # visualize all the hyperparameters
    pprint(glmsingle_obj.params)

    sys.stdout.flush()
    
    start_time = time.time()
    if args.overwrite or not exists(outputdir):
        print(f'running GLMsingle... Outputdir exists: {exists(outputdir)} and is being overwritten: {args.overwrite}')
    
        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
            design,
            data,
            args.stimdur,
            args.tr,
            outputdir=outputdir)

    else:
        print(f'GLMsingle outputs already exists in directory:\n\t{outputdir}')

    sys.stdout.flush()
    elapsed_time = time.time() - start_time

    print(
        '\telapsed time: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )

    print('finished running GLMsingle')
    


if __name__ == '__main__':
    main()


