import mne
from mne.preprocessing import ICA
import os.path as op
import json
import argparse
from tools import files
import numpy as np
import pandas as pd

json_file = "pipeline_params.json"

# argparse input
des = "pipeline script"
parser = argparse.ArgumentParser(description=des)
parser.add_argument(
    "-f", 
    type=str, 
    nargs=1,
    default=json_file,
    help="JSON file with pipeline parameters"
)

parser.add_argument(
    "-n", 
    type=int, 
    help="id list index"
)

args = parser.parse_args()
params = vars(args)
json_file = params["f"]
subj_index = params["n"]

# read the pipeline params
with open(json_file) as pipeline_file:
    pipeline_params = json.load(pipeline_file)

# paths
data_path = pipeline_params["data_path"]
fs_path = op.join(data_path, "MRI")

subjs = files.get_folders_files(fs_path, wp=False)[0]
subjs.sort()
subjs = [i for i in subjs if "fsaverage" not in i]
subj = subjs[subj_index]

meg_subj_path = op.join(data_path,"MEG", subj)
beh_subj_path = op.join(data_path,"BEH", subj)

verb=False

if pipeline_params["downsample_convert_filter"]:
    raw_ds = files.get_folders_files(
        meg_subj_path,
        wp=True
    )[0]
    raw_ds = [i for i in raw_ds if ".ds" in i]

    for ix, raw_path in enumerate(raw_ds):
        raw = mne.io.read_raw_ctf(
            raw_path,
            preload=True,
            verbose=False
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )

        events = mne.find_events(
            raw,
            stim_channel="UPPT001"
        )

        raw, events = raw.copy().resample(
            pipeline_params["downsample_to"], 
            npad="auto", 
            events=events,
            n_jobs=-1,
        )
        print(subj, ix, "resampled")
        raw = raw.filter(
            0.1,
            80,
            picks=picks_meg,
            n_jobs=-1,
            method="fir",
            phase="minimum"
        )
        print(subj, ix, "filtered")
        raw_out_path = op.join(
            meg_subj_path,
            "raw-{}-raw.fif".format(str(ix).zfill(3))
        )
        events_out_path = op.join(
            meg_subj_path,
            "{}-eve.fif".format(str(ix).zfill(3))
        )

        ica_out_path = op.join(
            meg_subj_path,
            "{}-ica.fif".format(str(ix).zfill(3))
        )

        n_components = 50
        method = "extended-infomax"
        reject = dict(mag=4e-12)

        ica = ICA(
            n_components=n_components, 
            method=method
        )

        ica.fit(
            raw, 
            picks=picks_meg,
            reject=reject,
            verbose=verb
        )
        print(subj, ix, "ICA_fit")
        raw.save(raw_out_path, overwrite=True)
        mne.write_events(events_out_path, events)
        ica.save(ica_out_path)
        print(subj, ix, "saved")

if pipeline_params["apply_ICA"]:
    ica_json = files.get_files(
        meg_subj_path,
        "",
        "ica-rej.json"
    )[2][0]

    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif",
        wp=False
    )[2]

    comp_ICA_json_path = op.join(
        meg_subj_path,
        "{}-ica-rej.json".format(str(subj).zfill(3))
    )

    ica_files = files.get_files(
        meg_subj_path,
        "",
        "-ica.fif",
        wp=False
    )[2]
    
    with open(ica_json) as data:
        components_rej = json.load(data)

    for k in components_rej.keys():
        raw_path = op.join(
            meg_subj_path,
            files.items_cont_str(raw_files, k, sort=True)[0]
        )
        ica_path = op.join(
            meg_subj_path,
            files.items_cont_str(ica_files, k, sort=True)[0]
        )
        
        raw = mne.io.read_raw_fif(
            raw_path,
            preload=True,
            verbose=verb
        )

        ica = mne.preprocessing.read_ica(ica_path)
        raw_ica = ica.apply(
            raw,
            exclude=components_rej[k]
        )

        raw_ica.save(
            raw_path,
            fmt="single",
            split_size="2GB",
            overwrite=True
        )

        print(raw_path)

if pipeline_params["epochs"]:
    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif",
        wp=True
    )[2]
    raw_files.sort()

    eve_files = files.get_files(
        meg_subj_path,
        "",
        "-eve.fif",
        wp=True
    )[2]
    eve_files.sort()

    all_files = zip(raw_files, eve_files)
    for raw_file, event_file, in all_files:

        raw = mne.io.read_raw_fif(
            raw_file,
            preload=False,
            verbose=verb
        )

        picks_meg = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            eog=False, 
            ecg=False, 
            ref_meg=False
        )

        events = mne.read_events(
            event_file
        )

        tmin, tmax = (-0.2, 0.5)
        baseline = (tmin, 0.0)
        epochs = mne.Epochs(
            raw,
            events,
            picks=picks_meg,
            tmin=tmin,
            tmax=tmax,
            detrend=1,
            baseline=baseline
        )
        epochs.apply_baseline(baseline)

        all_file_out = op.join(
            meg_subj_path,
            "all-{}-epo.fif".format(op.split(raw_file)[1].split("-")[1])
        )
        epochs.save(all_file_out)

if pipeline_params["fwd_solution"]:
    src = mne.setup_source_space(
        subject=subj, 
        subjects_dir=fs_path, 
        spacing="ico5", 
        add_dist=False
    )
    conductivity = (0.3,)
    model = mne.make_bem_model(
        subject=subj,
        ico=5,
        conductivity=conductivity,
        subjects_dir=fs_path
    )

    bem = mne.make_bem_solution(model)

    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif"
    )[2]
    raw_files.sort()

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()

    trans_file = op.join(
        meg_subj_path,
        "{}-trans.fif".format(subj)
    )

    all_files = zip(raw_files, epo_files)
    for raw_file, epo_file in all_files:
        file_id = op.split(raw_file)[1].split("-")[1]

        fwd_out = op.join(
            meg_subj_path,
            "fwd-{}-fwd.fif".format(file_id)
        )

        fwd = mne.make_forward_solution(
            raw_file,
            trans=trans_file,
            src=src,
            bem=bem,
            meg=True,
            eeg=False,
            mindist=5.0,
            n_jobs=-1
        )
        
        mne.write_forward_solution(
            fwd_out, 
            fwd, 
            verbose=verb
        )

        print(fwd_out)

if pipeline_params["cov_matrix"]:
    raw_files = files.get_files(
        meg_subj_path,
        "raw",
        "-raw.fif"
    )[2]
    raw_files.sort()
    for raw_file in raw_files:
        file_id = op.split(raw_file)[1].split("-")[1]

        cov_mx_out = op.join(
            meg_subj_path,
            "mx-{}-cov.fif".format(file_id)
        )

        raw = mne.io.read_raw_fif(
            raw_file, 
            preload=True,
            verbose=verb
        )

        picks = mne.pick_types(
            raw.info, 
            meg=True, 
            eeg=False, 
            stim=False, 
            eog=False, 
            ref_meg="auto", 
            exclude="bads"
        )

        noise_cov = mne.compute_raw_covariance(
            raw, 
            method="auto", 
            rank=None,
            picks=picks,
            n_jobs=-1
        )

        noise_cov.save(
            cov_mx_out
        )

        print(cov_mx_out)

if pipeline_params["inv_operator"]:

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()
    cov_files = files.get_files(
        meg_subj_path,
        "mx",
        "-cov.fif"
    )[2]
    cov_files.sort()
    fwd_files = files.get_files(
        meg_subj_path,
        "fwd",
        "-fwd.fif"
    )[2]
    fwd_files.sort()

    all_files = zip(epo_files, cov_files, fwd_files)

    for epo_path, cov_path, fwd_path in all_files:
        file_id = op.split(epo_path)[1].split("-")[1]

        inv_out = op.join(
            meg_subj_path,
            "inv-{}-inv.fif".format(file_id)
        )

        fwd = mne.read_forward_solution(fwd_path)

        cov = mne.read_cov(cov_path)

        epochs = mne.read_epochs(epo_path)

        inv = mne.minimum_norm.make_inverse_operator(
            epochs.info,
            fwd,
            cov,
            loose=0.2,
            depth=0.8
        )

        mne.minimum_norm.write_inverse_operator(
            inv_out,
            inv
        )

        print(inv_out)


if pipeline_params["compute_inverse"][0]:
    method_dict = {
        "dSPM": (8, 12, 15),
        "sLORETA": (3, 5, 7),
        "eLORETA": (0.75, 1.25, 1.75)
    }

    method = pipeline_params["compute_inverse"][1]
    snr = 3.
    lambda2 = 1. / snr ** 2
    lims = method_dict[method]

    epo_files = files.get_files(
        meg_subj_path,
        "all",
        "-epo.fif"
    )[2]
    epo_files.sort()

    inv_files = files.get_files(
        meg_subj_path,
        "inv",
        "-inv.fif"
    )[2]
    inv_files.sort()

    all_files = zip(epo_files, inv_files)

    for epo_path, inv_path in all_files:
        file_id = op.split(epo_path)[1].split("-")[1]

        stc_out = op.join(
            meg_subj_path,
            "stc-{}"
        )

        epo = mne.read_epochs(
            epo_path,
            verbose=verb,
            preload=True
        )

        epo = epo.average()

        inv = mne.minimum_norm.read_inverse_operator(
            inv_path,
            verbose=verb
        )

        stc = mne.minimum_norm.apply_inverse(
            epo,
            inv,
            lambda2,
            method=method,
            pick_ori=None,
            verbose=True
        )

        # stc.save(stc_out)
