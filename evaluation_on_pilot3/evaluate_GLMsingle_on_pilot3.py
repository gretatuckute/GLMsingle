# # Find optimal GLMsingle parameters based on control expt pilot 3
# 
# ---------------------
# 
# ##### GLMsingle is new tool that provides efficient, scalable, and accurate single-trial fMRI response estimates.
# 
# Input: swr* files OR wr* files
# Parameters to test: HRFs, different number of PCs, fracridge parameter.
# 
# Every single trial is its own condition (i.e., 1,000 unique conditions), which is the item_id. So, if the first sentence presented in session 1, run 1, was item_id 994, then that is the condition.
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
    
    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    # ### Set paths and download the example dataset
    user = getpass.getuser()
    print(f'Running as user {user}')
    if user != 'gt':
        root = '/om5/group/evlab/u/gretatu/GLMsingle/'
    else:
        root = '/Users/gt/om5/gretatu/GLMsingle/'

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
    stimdur = 2
    tr = 2
    
    # create directory for saving data
    datadir = join(root, 'input_neural_data')

    # create directory for saving outputs from example 1
    outputdir = join(root, 'output_glmsingle', f'output_glmsingle_preproc-{preproc}_pcstop{pcstop}_fracs-{fracs}')
    designdir = join(root, 'design_matrices')   # set design matrix directory
    logdir = join(root, 'logs')

    if user != 'gt' and not args.verbose:
        date = datetime.datetime.now().strftime("%Y%m%d-%T")
        sys.stdout = open(join(logdir, f'out_{preproc}_pcstop{pcstop}_fracs-{fracs}_{date}.log'), 'a+')
        
    print(f'Preprocessing pipeline: {preproc} with {pcstop} PCs and {fracs} fracridge')
    print(f'\nInput data dir: {datadir}')
    print(f'\nSave output dir: {outputdir}')
    print(f'\nDesign matrices dir: {designdir}')
    print(f'\nLog dir: {logdir}\n')

    ### Organize BOLD data, design matrices, metadata
    UID = '853'
    SESSIONS = ['FED_20211008a_3T1_PL2017', 'FED_20211013b_3T1_PL2017']
    if args.test:
        print(f'Running in test mode. Only using one session: {SESSIONS[:1]}')
        SESSIONS = SESSIONS[:1]
    n_trs = 168
    design_matrix_name = 'design_matrices_UID-853_SESSION-FED_20211008a_3T1_PL2017-FED_20211013b_3T1_PL2017_FL-control_tr1_20220109_singletrial.pkl'

    data = []
    for s in (SESSIONS):
        print(f'Loading data from session {s}')
        for i, f in enumerate(sorted(os.listdir(join(datadir, UID, s)))):
            if args.test:
                if i > 0:
                    break
            if f.startswith(preproc):
                print(f'Loaded file: {f}')
                file = np.array(nib.load(join(datadir, UID, s, f)).dataobj)
                assert (file.shape[3] == n_trs)
                data.append(file)

    # get shape of data volume (XYZ) for convenience
    xyz = data[0].shape[:3]
    xyzt = data[0].shape

    print(f'Number of runs in data: {len(data)}.\nShape of Images (brain XYZ and TR): {data[0].shape}')
    design = pd.read_pickle(join(designdir, design_matrix_name))
    if args.test:
        design = [design[0]]
    
    print(f'Number of runs in design matrix: {len(design)}, each with shape: {design[0].shape} [TR; cond]')

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
    print(f'The stimulus duration is {stimdur} seconds (TR={tr})\n')
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
    # opt['sessionindicator'] = np.repeat([1, 2], 10)

    # add changing parameters
    opt['pcstop'] = pcstop
    opt['fracs'] = fracs

    # running python GLMsingle involves creating a GLM_single object
    # and then running the procedure using the .fit() routine
    glmsingle_obj = GLM_single(opt)

    # visualize all the hyperparameters
    pprint(glmsingle_obj.params)

    sys.stdout.flush()

    # this example saves output files to the folder  "example1outputs/GLMsingle"
    # if these outputs don't already exist, we will perform the time-consuming call to GLMsingle;
    # otherwise, we will just load from disk.

    start_time = time.time()

    if args.overwrite or not exists(outputdir):
        print(f'running GLMsingle... Outputdir exists: {exists(outputdir)} and is being overwritten: {args.overwrite}')
    
        # run GLMsingle
        results_glmsingle = glmsingle_obj.fit(
            design,
            data,
            stimdur,
            tr,
            outputdir=outputdir)
    
        # we assign outputs of GLMsingle to the "results_glmsingle" variable.
        # note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model,
        # where all images are treated as the same condition. these estimates
        # could be potentially used to find cortical areas that respond to
        # visual stimuli. we want to compare beta weights between conditions
        # therefore we are not going to include the ONOFF betas in any analyses of
        # voxel reliability

    else:
        print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir}')
    
        # load existing file outputs if they exist
        results_glmsingle = dict()
        results_glmsingle['typea'] = np.load(join(outputdir, 'TYPEA_ONOFF.npy'), allow_pickle=True).item()
        results_glmsingle['typeb'] = np.load(join(outputdir, 'TYPEB_FITHRF.npy'), allow_pickle=True).item()
        results_glmsingle['typec'] = np.load(join(outputdir, 'TYPEC_FITHRF_GLMDENOISE.npy'), allow_pickle=True).item()
        results_glmsingle['typed'] = np.load(join(outputdir, 'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                             allow_pickle=True).item()

    sys.stdout.flush()
    elapsed_time = time.time() - start_time

    print(
        '\telapsed time: ',
        f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
    )

    print('finished running GLMsingle')
    


if __name__ == '__main__':
    main()


