import pickle
import json

"""
IO functions for moedal

"""

def save_json(data,filepath):
    with open(filepath,'w') as outfile:
        print('JSON saved :',filepath)
        json.dump(data,outfile)

def load_json(filepath):
    with open(filepath) as json_file:
        print('JSON loaded :',filepath)
        return json.load(json_file)

def pickle_load(filepath):
    # file open must be binary 'rb'
    with open(filepath,'rb') as pickle_file:
        print('Data loaded :',filepath)
        return pickle.load(pickle_file)

def pickle_dump(data,filepath):
    # file open must be binary 'wb', doesnt need extension
    with open(filepath,'wb') as outfile:
        print('Data saved :',filepath)
        pickle.dump(data,outfile)

"""
 Json will not serialise numpy floats / ints / arrays
 Following lambda's can be used to map interger conversion
 onto tuples / arrays
"""        
listmap = lambda x, func : list(map(func,x))
toint = lambda x : listmap(x,int)
