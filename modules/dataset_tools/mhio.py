import pickle
import json

try:
    from modules.pipeline_paths import resolve_pipeline_path
except ModuleNotFoundError:  # Support notebooks that add only repo/modules to sys.path.
    from pipeline_paths import resolve_pipeline_path

"""
IO functions for moedal

"""

def save_json(data,filepath):
    filepath = resolve_pipeline_path(filepath, for_write=True)
    with open(filepath,'x') as outfile:
        print('JSON saved :',filepath)
        json.dump(data,outfile)

def load_json(filepath):
    filepath = resolve_pipeline_path(filepath, must_exist=True)
    with open(filepath) as json_file:
        print('JSON loaded :',filepath)
        return json.load(json_file)

def pickle_load(filepath):
    # file open must be binary 'rb'
    filepath = resolve_pipeline_path(filepath, must_exist=True)
    with open(filepath,'rb') as pickle_file:
        print('Data loaded :',filepath)
        return pickle.load(pickle_file)

def pickle_dump(data,filepath):
    # file open must be binary 'wb', doesnt need extension
    filepath = resolve_pipeline_path(filepath, for_write=True)
    with open(filepath,'xb') as outfile:
        print('Data saved :',filepath)
        pickle.dump(data,outfile)

"""
 Json will not serialise numpy floats / ints / arrays
 Following lambda's can be used to map interger conversion
 onto tuples / arrays
"""        
listmap = lambda x, func : list(map(func,x))
toint = lambda x : listmap(x,int)
