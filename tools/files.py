import os
from itertools import compress
from collections import OrderedDict
import json


def get_folders_files(path, wp=True):
    """
    Returns lists of files and folders from directory
    path - folder address
    wp - BOOL, list with path
    """
    files = []
    folders = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        folders.extend(dirnames)
        break
    if wp:
        files = [path + os.sep + i for i in files]
        folders = [path + os.sep + i for i in folders]

    return (folders, files)


def get_files(path, prefix, extension, wp=True):
    """
    Returns lists of files within folder with specific
    prefix, extension and both.
    path - folder address
    wp - BOOL, list with path
    prefix - string, beginning of the filename
    extension - string, extension w/o dot
    """
    ext = []
    prfx = []
    intersect = []
    files = get_folders_files(path, wp=False)
    for i in files[1]:
        if i.startswith(prefix):
            prfx.append(i)
        if i.endswith(extension):
            ext.append(i)
        if i.endswith(extension) and i.startswith(prefix):
            intersect.append(i)
    if wp:
        ext = [path + os.sep + i for i in ext]
        prfx = [path + os.sep + i for i in prfx]
        intersect = [path + os.sep + i for i in intersect]

    return (ext, prfx, intersect)

def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def items_cont_str(input_list, string, sort=False):
    """
    returns a list of items which contain a given string
    optionally sorted
    """
    output_list = [string in i for i in input_list]
    output_list = list(compress(input_list, output_list))
    if sort:
        output_list.sort()
    return output_list

   
def update_key_value(file, key, value):
    """
    Function to update a key value in a JSON file. If passed
    key is not in the JSON file, it is going to be appended 
    at the end of the file.
    """
    with open(file, "r") as json_file:
        data = json.load(json_file, object_pairs_hook=OrderedDict)
        data[key] = value
    
    with open(file, "w") as json_file:
        json.dump(data, json_file, indent=4)


def dump_the_dict(file, dictionary):
    """
    Function dumps dictionary to a JSON file.
    """
    with open(file, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)