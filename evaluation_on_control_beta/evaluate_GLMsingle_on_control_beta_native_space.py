"""Evaluate single-trial responses for control beta experiment.

Every single trial is its own condition (e.g., 1,000 unique conditions), which is the item_id.
"""

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

from utils import *


# warnings.filterwarnings('ignore')
SUBJECTSDIR = '/nese/mit/group/evlab/u/Shared/SUBJECTS/'


def str2none(v):
    """If string is 'None', return None. Else, return the string"""
    if v is None:
        print(f'Already None: {v}')
        return v
    if v.lower() in ('none'):
        print(f'String arg - None: {v}')
        return None
    else:
        return v

def run_id_to_nii(subject_id: str,
                  OUTPUT_STIMSET_DIR: str,
                  SUBJECTSDIR: str,
                  stimset_name: str,
                  expected_duration: int,
                  tr: int,
                  n_runs: int,
                  preproc_config: str = 'native_space',
                  save: bool = True,
                  overwrite: bool = False):
    """
    Add dicom information to stimset.
    Get dicom info and paths to preprocessed nii files.
    """

    # Print all args
    print(f'Running run_id_to_dicom with the following args: {locals()}\n')

    # Load the stimset (has both sessions)
    df_stimset = pd.read_csv(join(OUTPUT_STIMSET_DIR, f'stimset_{stimset_name}.csv'))
    df_stimset_orig = df_stimset.copy(deep=True)

    # Get the session id for this subject
    session_ids = d_UID_to_session_list[int(subject_id)]

    # Get unique sessions
    unique_uid_sessions = df_stimset['uid_session'].unique()
    # just get the last part (after the UID) and assert that it matches the session_ids
    unique_sessions = ['_'.join(x.split('_')[1:]) for x in unique_uid_sessions]
    assert session_ids == unique_sessions

    lst_df_dicom_across_sess = []
    for session_id in session_ids:
        # session_id = 'FED_20220414c_3T1'

        # Get a stimset that only has the current session
        df_stimset_sess = df_stimset_orig[df_stimset_orig['uid_session'].str.contains(session_id)]

        # Generate a simple savestr:
        savestr = f'{stimset_name}_{subject_id}_session-{session_id}'

        # Get the uid session
        uid_session = f'{subject_id}_{session_id}'

        # Load info from data.cfg file
        data_cfg_path = join(SUBJECTSDIR,
                             f'{uid_session}_PL2017')
        data_cfg = pd.read_csv(join(data_cfg_path,
                                    'data.cfg'),
                               header=None)

        # Reformat data.cfg -- map dicoms to functionals
        # Get dicoms (all lines below #dicoms)
        dicoms = data_cfg[data_cfg[0].str.contains('dicoms')]
        # Omit first one row name
        dicoms = dicoms[1:].values

        # Get functional numbers (line below #functionals)
        functionals_idx = data_cfg[data_cfg[0].str.contains('functionals')].index[0]
        functionals = data_cfg.iloc[functionals_idx + 1].values
        # Remove trailing whitespace
        functionals = [f.strip() for f in functionals]
        # convert numpy string array to list of ints
        functionals = functionals[0].split(' ')
        functionals = [int(i) for i in functionals]

        # Read dicom summary
        dicom_summary_path = join(SUBJECTSDIR,
                                  f'{uid_session}_PL2017',
                                  'dicom_summary.csv')
        dicom_summary = pd.read_csv(dicom_summary_path, index_col=False)

        # Get RUN_NUM for all rows that have IPS=expected_IPS
        expected_IPS = int(expected_duration / tr)

        run_nums = dicom_summary.query(f'IPS == {expected_IPS}')['RUN_NUM'].values
        assert len(run_nums) == n_runs
        assert all([i in functionals for i in run_nums])  # assert that all runs nums exist in functionals

        # Get the idx of the runs from functionals, but note that it is 1-indexed
        run_nums_idx = [functionals.index(i) + 1 for i in run_nums]

        # In dicoms, obtain the dicom string if it is in the run_nums
        dicoms_in_run_nums = []  # for IPS of interest
        dicom_nums = []
        dicom_ids = []
        for dicom in dicoms:
            dicom_num = dicom[0].split('-')[-2]
            dicom_id = '-'.join(dicom[0].split('.')[-2].split('/')[-1].split('-')[
                                :-1])  # Get identifier consisting of the random number and then the dicom image number

            if int(dicom_num) in run_nums:
                dicoms_in_run_nums.append(dicom[0])
                dicom_nums.append(dicom_num)
                dicom_ids.append(dicom_id)

        df_dicom = pd.DataFrame({'dicomnumber': dicom_nums,
                                 'dicomid': dicom_ids,
                                 'dicom_path': dicoms_in_run_nums})

        # Get the dicomids from stimset
        stimset_dicomids = df_stimset_sess.loc[df_stimset_sess['uid_session'] == uid_session, 'dicomid'].unique()

        # Get the run numbers from the control runs (1-10)
        crit_expt_run_id = np.unique(df_stimset_sess['run_id'].values)
        assert all(np.diff(crit_expt_run_id) == 1)  # make sure they are sorted and ascending

        df_dicom['run_id'] = crit_expt_run_id
        df_dicom['run_nums_idx_functionals'] = run_nums_idx  # Which number it has in the preprocessed nii files
        df_dicom['uid_session'] = uid_session

        lst_df_dicom_across_sess.append(df_dicom)

    df_dicom_across_sess = pd.concat(lst_df_dicom_across_sess)

    # Add to stimset and match to the run_id
    df_stimset = df_stimset.merge(df_dicom_across_sess, on='run_id', how='left')

    # Assert whether _x and _y columns are the same and only keep one
    xy_cols = [col for col in df_stimset.columns if col.endswith('_x')]
    for col in xy_cols:
        if col == 'dicomnumber_x':
            # make both int
            assert df_stimset[col].astype(int).equals(df_stimset[col[:-2] + '_y'].astype(int))
        else:
            assert df_stimset[col].equals(df_stimset[col[:-2] + '_y'])
        df_stimset = df_stimset.drop(columns=[col])
        # Rename the y column to the original name
        df_stimset = df_stimset.rename(columns={col[:-2] + '_y': col[:-2]})

    # Create the path to the preprocessed nii files

    # E.g., rfunc_run-03_bold.nii
    df_stimset['nii_r_path'] = [join(SUBJECTSDIR,
                                    f'{df_stimset.loc[idx, "uid_session"]}_PL2017',
                                    preproc_config,
                                    'func',
                                    f'rfunc_run-{str(run_idx).zfill(2)}_bold.nii') for idx, run_idx in
                                enumerate(df_stimset['run_nums_idx_functionals'])]


    # E.g., wrfunc_run-03_bold.nii
    df_stimset['nii_wr_path'] = [join(SUBJECTSDIR,
                                        f'{df_stimset.loc[idx, "uid_session"]}_PL2017',
                                      preproc_config,
                                      'func',
                                        f'wrfunc_run-{str(run_idx).zfill(2)}_bold.nii') for idx, run_idx in
                                     enumerate(df_stimset['run_nums_idx_functionals'])]

    # E.g., swrfunc_run-03_bold.nii
    df_stimset['nii_swr_path'] = [join(SUBJECTSDIR,
                                        f'{df_stimset.loc[idx, "uid_session"]}_PL2017',
                                       preproc_config,
                                       'func',
                                        f'swrfunc_run-{str(run_idx).zfill(2)}_bold.nii') for idx, run_idx in
                                     enumerate(df_stimset['run_nums_idx_functionals'])]

    # Assert that these files actually exist
    if preproc_config == 'native_space':
        assert all([os.path.exists(i) for i in df_stimset['nii_r_path']])
    if preproc_config == 'DefaultMNI_PlusStructural':
        assert all([os.path.exists(i) for i in df_stimset['nii_r_path']])
        assert all([os.path.exists(i) for i in df_stimset['nii_wr_path']])
        assert all([os.path.exists(i) for i in df_stimset['nii_swr_path']])

    df_stimset['expected_IPS'] = expected_IPS

    # Assert that nothing weird happened with e.g., item_id indexing
    assert df_stimset_orig.item_id.equals(df_stimset.item_id)

    # Save (take the original name and suffix _wdicom)
    if save:
        fname = join(OUTPUT_STIMSET_DIR, stimset_name, f'stimset_{stimset_name}_wnii.csv')

        if not os.path.exists(fname) or overwrite:
            df_stimset.to_csv(fname, index=False)
            print(f'Saved {fname} to {OUTPUT_STIMSET_DIR}')
        elif os.path.exists(fname) and not overwrite:
            print(f'File {fname} already exists. Set overwrite=True to overwrite it.')
        else:
            print(f'File {fname} already exists. Set save=True to save it.')

    return df_stimset



def main(raw_args=None):
    # Mapping specific
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--UID', default='865', type=str, help='UID str')
    parser.add_argument('--FL', default='gs', type=str, help='FL')
    parser.add_argument('--stimdur', default=2, type=int, help='Stimulus duration in seconds')
    parser.add_argument('--tr', default=2, type=int, help='TR sampling rate')
    parser.add_argument('--preproc', default='r', type=str,
                        help='Which preprocessing pipeline to use. Default is swr.')
    parser.add_argument('--pcstop', default=5, type=int,
                        help='How many PCs to remove')
    parser.add_argument('--fracs', default=0.05, type=float,
                        help='Fraction of ridge regularization to use')
    parser.add_argument('--test', default=False, type=bool,
                        help='Whether to run test mode and only use one run for testing')
    parser.add_argument('--verbose', default=True, type=bool,
                        help='Whether to print output and not create a log file')
    parser.add_argument('--overwrite', default=True, type=bool,
                        help='Whether to overwrite results in case outputdir already exists')
    parser.add_argument('--external_output_root', default='/nese/mit/group/evlab/u/gretatu/GLMsingle/', type=str2none,
                        help='If not None, supply a path to a directory to save outputs to')
    args = parser.parse_args(raw_args)

    ### Set paths ###
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

    ### Arguments to change in GLM ###
    pcstop = -args.pcstop
    if pcstop == 0:
        pcstop = '-0'  # make sure the string names are correct!
    fracs = args.fracs

    # data args
    preproc = args.preproc

    # create directory for saving data
    if args.external_output_root is None:
        output_root = join(root, 'output_glmsingle')
    else:
        output_root = join(args.external_output_root, 'output_glmsingle')

    outputdir = join(output_root,
                     f'output_glmsingle_preproc-{preproc}_pcstop{pcstop}_fracs-{fracs}_UID-{args.UID}')
    designdir = join(root, 'design_matrices')  # set design matrix directory
    stimsetdir = join(designdir, 'associated_stimsets')  # set stimset directory
    logdir = join(root, 'logs')

    if user != 'gt' and not args.verbose:
        date = datetime.datetime.now().strftime("%Y%m%d-%T")
        sys.stdout = open(
            join(logdir, f'eval_control_beta_native_space_{preproc}_pcstop{pcstop}_fracs-{fracs}_UID-{args.UID}_{date}.log'), 'a+')

    print('*' * 40)
    print(vars(args))
    print('*' * 40)

    print(f'Preprocessing pipeline: {preproc} with {pcstop} PCs and {fracs} fracridge')
    print(f'\nSave output dir: {outputdir}')
    print(f'\nDesign matrices dir: {designdir}')
    print(f'\nLog dir: {logdir}\n')

    if pcstop == '-0':
        pcstop = -0  # revert back

    ### Organize BOLD data, design matrices, metadata
    SESSION_TO_INCLUDE = d_UID_to_session_list[int(args.UID)]
    session_str = '-'.join(SESSION_TO_INCLUDE)

    if not SESSION_TO_INCLUDE[0].endswith('PL2017'):
        SESSIONS = [x + '_PL2017' for x in SESSION_TO_INCLUDE]
    if args.test:
        print(f'Running in test mode. Only using one session: {SESSION_TO_INCLUDE[:1]}')
        SESSIONS = SESSION_TO_INCLUDE[:1]

    # Design matrix
    load_str = f'UID-{args.UID}_SESSION-{session_str}'
    design = pd.read_pickle(join(designdir, f'design_matrices_{load_str}.pkl'))

    # Associated stimset
    stimset = pd.read_csv(join(stimsetdir, f'stimset_{load_str}.csv'))
    # want to ensure that the loaded neural data matches the stimset and design matrix

    """In the new el pipeline, we can't load the preprocessed nii files via their dicom number. We need to find a 
    a given functional run's nii file by its 1-indexed number (ranging between 1 and number of functional runs in that session)
    """

    # Get the dicomid to func run num mappings
    stimset_w_nii = run_id_to_nii(subject_id=args.UID,
                          OUTPUT_STIMSET_DIR=stimsetdir,
                          SUBJECTSDIR=SUBJECTSDIR,
                          stimset_name=load_str,
                          expected_duration=336,
                          tr=2,
                          n_runs=10,
                          save=False, overwrite=False)


    data = []
    session_indicators = []

    # Load each unique dicom image and retain the order
    _, idx = np.unique(stimset_w_nii[f'nii_{args.preproc}_path'].values, return_index=True)
    images_of_interest = stimset_w_nii[f'nii_{args.preproc}_path'].values[np.sort(idx)]

    for i, s in enumerate(images_of_interest):
        print(f'Loading data from session UID: {args.UID}, dicom image: {s}\n')
        if args.test:
            if i == 7:
                break

        # s = '/nese/mit/group/evlab/u/Shared/SUBJECTS/865_FED_20220414c_3T1_PL2017/native_space/func/rfunc_run-03_bold.nii'
        file = np.array(nib.load(s).dataobj)
        assert (file.shape[3] == design[i].shape[0])
        data.append(file)
        session_indicator = stimset_w_nii.loc[stimset_w_nii[f'nii_{args.preproc}_path'] == s, 'sessionindicator'].values[0]
        session_indicators.append(session_indicator)

    # get shape of data volume (XYZ) for convenience
    xyz = data[0].shape[:3]
    xyzt = data[0].shape

    print(f'Number of runs in data: {len(data)}.\nShape of Images (brain XYZ and TR): {data[0].shape}')
    if args.test:
        design = design[:7]

    print(
        f'Number of runs in design matrix: {len(design)}, with unique number of TRs across runs: {np.unique([x.shape[0] for x in design])}\n'
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
        plt.ylabel('time (TR)', fontsize=16)

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
