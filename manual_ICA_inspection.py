import matplotlib
import mne.gui
import os.path as op
import json
import argparse
from tools import files

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

parser.add_argument(
    "-file", 
    type=int, 
    help="file list index"
)

args = parser.parse_args()
params = vars(args)
json_file = params["f"]
subj_index = params["n"]
file_index = params["file"]

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


raw_files = files.get_files(
    meg_subj_path,
    "raw",
    "-raw.fif"
)[2]
raw_files.sort()

raw_file = raw_files[file_index]

comp_ICA_json_path = op.join(
    meg_subj_path,
    "{}-ica-rej.json".format(str(subj).zfill(3))
)

ica_files = files.get_files(
    meg_subj_path,
    "",
    "-ica.fif"
)[2]
ica_files.sort()

ica_file = ica_files[file_index]

if not op.exists(comp_ICA_json_path):
    json_dict = {
        i[-11:-8]: [] for i in raw_files
    }
    files.dump_the_dict(comp_ICA_json_path, json_dict)

raw = mne.io.read_raw_fif(raw_file , preload=True, verbose=False)

set_ch = {'EEG057-3305':'eog', 'EEG058-3305': 'eog'}
raw.set_channel_types(set_ch)

ica = mne.preprocessing.read_ica(ica_file)

eog_ix, eog_scores = ica.find_bads_eog(
    raw, 
    threshold=3.0, 
    l_freq=1, 
    h_freq=10, 
    verbose=False
)
eog_ix.sort()
print(subj)
print(eog_ix)

ica.plot_scores(eog_scores, exclude=eog_ix)

ica.plot_sources(raw)