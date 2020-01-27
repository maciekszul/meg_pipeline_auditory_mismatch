import mne.gui
import os.path as op
from tools import files
import json
import sys

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
    wp=True
)[2]
raw_paths.sort()

raw_path = raw_paths[file_index]
print(raw_path)
print(subject)

mne.gui.coregistration(
    subjects_dir=subjects_dir,
    subject=subject,
    inst=raw_path
)