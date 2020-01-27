import time
named_tuple = time.localtime() # get struct_time
time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
print("start:", time_string)

import mne
from mne.preprocessing import ICA
import os.path as op
import json
from tools import files
import numpy as np
import sys


# parsing command line arguments
try:
    subj_index = int(sys.argv[1])
except:
    print("incorrect arguments")
    sys.exit()

try:
    json_file = sys.argv[2]
    print(json_file)
except:
    json_file = "pipeline_params.json"
    print(json_file)

# open json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

# prepare paths
data_path = parameters["path"]
subjects_dir = parameters["freesurfer"]

# subjects
subjs = files.get_folders_files(subjects_dir, wp=False)[0]
subjs = [i for i in subjs if "00" in i]
subjs.sort()
subj = subjs[subj_index]

# processing parameters
down_freq = parameters["downsample"]
chpi = [
    "HLC0011", "HLC0012", "HLC0013", 
    "HLC0021", "HLC0022", "HLC0023",
    "HLC0031", "HLC0032", "HLC0033"
]

subj_path = op.join(data_path, "MEG", subj)

if parameters["convert_downsample_filter"]:
    all_ds = files.get_folders_files(
        subj_path,
        wp=True
    )[0]
    all_ds = [i for i in all_ds if ".ds" in i]
    all_ds.sort()
    for ix, ds in enumerate(all_ds):
        print(ds)
        # output paths
        raw_out = op.join(
            subj_path,
            "raw-{0}-raw.fif".format(str(ix).zfill(3))
        )

        eve_out = op.join(
            subj_path,
            "events-{0}-eve.fif".format(str(ix).zfill(3))
        )

        ica_out = op.join(
            subj_path,
            "ica-{0}-ica.fif".format(str(ix).zfill(3))
        )

        # load data
        raw = mne.io.read_raw_ctf(
            ds,
            preload=True,
            clean_names=True, 
            system_clock="ignore"
        )

        # set channel categories

        set_ch = {
            "EEG057":"eog", 
            "EEG058": "eog", 
            "UPPT001": "stim"
            }
        raw.set_channel_types(set_ch)

        events = mne.find_events(
            raw
        )

        # trimming the raws
        raw_sfreq = raw.info["sfreq"]
        try:
            crop_min = events[0][0] / raw_sfreq - 2
            crop_max = events[-1][0] / raw_sfreq + 2
            raw.crop(tmin=crop_min, tmax=crop_max)
        except:
            crop_min = events[0][0] / raw_sfreq - 2
            crop_max = events[-1][0] / raw_sfreq + 1
            raw.crop(tmin=crop_min, tmax=crop_max)
        
        filter_picks = mne.pick_types(
            raw.info,
            meg=True,
            misc=False,
            stim=False,
            eog=True
        )

        raw = raw.filter(
            0.01,
            None,
            method="fir",
            phase="zero-double",
            n_jobs=-1,
            picks=filter_picks
        )
        

        raw = raw.notch_filter(
            np.arange(50, 251, 50),
            picks=filter_picks,
            filter_length="auto",
            method="fir",
            n_jobs=-1,
            phase="zero-double"
        )

        raw = raw.filter(
            None,
            120,
            method="fir",
            phase="zero-double",
            n_jobs=-1,
            picks=filter_picks
        )

        raw, events = raw.copy().resample(
            down_freq, 
            npad="auto", 
            events=events,
            n_jobs=-1,
        )

        # ICA
        n_components = 50
        method = "fastica"
        max_iter = 10000

        ica = ICA(
            n_components=n_components, 
            method=method,
            max_iter=max_iter
        )

        ica.fit(
            raw
        )

        # save the files
        raw.save(raw_out)
        ica.save(ica_out)
        mne.write_events(eve_out, events)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("converting, filtering and downsampling done:", time_string)

# MANUAL ICA COMPONENT INSPECTION use ICA_inspection.py

if parameters["apply_ica_epoch"]:
    raws = files.get_files(
        subj_path,
        "raw",
        "-raw.fif"
    )[0]
    raws.sort()

    icas = files.get_files(
        subj_path,
        "ica",
        "-ica.fif"
    )[0]
    icas.sort()

    evts = files.get_files(
        subj_path,
        "events",
        "-eve.fif"
    )[0]
    evts.sort()

    components_file_path = op.join(
        subj_path,
        "rejected-components.json"
    )

    with open(components_file_path) as data:
        components_rej = json.load(data)

    all_files = list(zip(raws, icas, evts))

    for raw_file, ica_file, events_path in all_files:

        raw = mne.io.read_raw_fif(
            raw_file,
            preload=True
        )
        file_n = raw_file.split("/")[-1]
        comp = components_rej[file_n]
        ica = mne.preprocessing.read_ica(ica_file)
        events = mne.read_events(events_path)
        raw = ica.apply(
            raw,
            exclude=comp
        )

        filter_picks = mne.pick_types(
            raw.info,
            meg=True,
            ref_meg=True,
            stim=False,
            eog=False,
            misc=False
        )

        raw = raw.filter(
            None,
            45,
            method="fir",
            phase="zero-double",
            n_jobs=-1,
            picks=filter_picks
        )

        epo_file = op.join(
            subj_path,
            "lp45-{}-epo.fif".format(file_n.split("-")[1])
        )

        onsets = mne.pick_events(events, exclude=[0])
        epochs =mne.Epochs(
            raw,
            onsets,
            tmin=-0.1,
            tmax=0.4,
            baseline=None
        )

        epochs.save(epo_file)

    named_tuple = time.localtime() # get struct_time
    time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
    print("applying ICA, filtering and epoching done:", time_string)

# use: ipython --gui=qt5 head_coregistration.py and head_coreg_check.py manually
# to produce -trans.fif files for source localisation
