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

raw_file = raw_files[file_index]


mne.gui.coregistration(
    subjects_dir=fs_path
)
