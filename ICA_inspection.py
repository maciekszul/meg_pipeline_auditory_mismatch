import mne
from mne.preprocessing import ICA
import os.path as op
import json
from tools import files
import numpy as np
import pandas as pd
import sys
import subprocess as sp

# parsing command line arguments
try:
    subj_index = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    json_file = sys.argv[3]
    print(json_file)
except:
    json_file = "pipeline_params.json"
    print(json_file)

try:
    file_index = int(sys.argv[2])
except:
    print("incorrect file index")
    sys.exit()

# open json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

# prepare paths
data_path = parameters["path"]
subjects_dir = parameters["freesurfer"]

subjects = files.get_folders_files(subjects_dir, wp=False)[0]
subjects.sort()
subjects = [i for i in subjects if "fsaverage" not in i]
subject = subjects[subj_index]

subject_meg = op.join(
    data_path,
    "MEG",
    subject
)

raw_paths = files.get_files(
    subject_meg,
    "raw-",
    "-raw.fif",
    wp=False
)[2]
raw_paths.sort()

ica_paths = files.get_files(
    subject_meg,
    "ica-",
    "-ica.fif",
    wp=False
)[2]
ica_paths.sort()

components_file_path = op.join(
    subject_meg,
    "rejected-components.json"
)

if not op.exists(components_file_path):
    json_dict = {
        i: [] for i in raw_paths
    }
    files.dump_the_dict(components_file_path, json_dict)

raw = mne.io.read_raw_fif(
    op.join(subject_meg, raw_paths[file_index]),
    preload=True,
    verbose=False
)

ica = mne.preprocessing.read_ica(
    op.join(subject_meg, ica_paths[file_index])
)

eog_ix, eog_scores = ica.find_bads_eog(
    raw, 
    threshold=3.0, 
    l_freq=1, 
    h_freq=10, 
    verbose=False
)

eog_ix.sort()

print(components_file_path)
print(raw_paths[file_index])

ica.plot_scores(eog_scores, exclude=eog_ix)

ica.plot_components()

ica.plot_sources(raw)

sp.Popen(
    ["gedit", str(components_file_path)], 
    stdout=sp.DEVNULL, 
    stderr=sp.DEVNULL
)