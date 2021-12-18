import os,sys,re,pickle
from numpy.lib.shape_base import split
import pandas as pd 
import numpy as np 

import json

from copy import deepcopy

import argparse

# ------------------------------------------------------------------------------------------------------------

def sort_parse_comma_separated_list(s):
  if isinstance(s, list):
    return sorted(s)
  return sorted( s.split(',') ) 


# ------------------------------------------------------------------------------------------------------------

def make_label(fname,label_type_dict): 
  

  keys = sorted (list (label_type_dict.keys() )) # ! sort will make back-track easier 

  label_array = []
  for k in keys : 
    onehot = [0] * len(label_type_dict[k])
    for i,x in enumerate(label_type_dict[k]): 
      if x in fname: 
        onehot[i] = 1
    # 
    label_array = label_array + onehot # concat list 

  assert np.sum(label_array) == len(label_type_dict) # must have unique name in each "type of label"
  
  return label_array
  

# ------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default=None)
parser.add_argument("--previous_path", type=str, default=None)
parser.add_argument("--normal_fin", type=str, default=None)
parser.add_argument("--syndrome_fin", type=str, default=None)
parser.add_argument("--age_bracket",type=sort_parse_comma_separated_list,default='2y,adolescence,olderadult,youngadult,youngchild')
parser.add_argument("--disease",type=sort_parse_comma_separated_list,default='WS,22q11DS,Controls,Normal')
parser.add_argument("--new_df_path", type=str, default=None)
parser.add_argument("--gender_csv", type=str, default=None)
parser.add_argument("--skip_normal", action='store_true', default=False)


# FairFace-aligned-60k-agegroup.csv
# age_ws_22q_controls_format.csv

args = parser.parse_args()

np.random.seed(seed=1)

# ! ! can reuse code from previous experiments 

# combine all normal faces
normal_fin = sorted ( args.normal_fin.split(',') ) 
normal_fin = pd.concat ( [pd.read_csv(os.path.join(args.previous_path,f)) for f in normal_fin] , ignore_index=True ) 

# read in train from last experiments, and append to normal 
syndrome_fin = pd.read_csv( os.path.join( args.previous_path,args.syndrome_fin ))
syndrome_fin = syndrome_fin[~syndrome_fin["label"].str.contains('Normal')].reset_index(drop=True)

# ! add gender to model ? why not. we need to fix csv input and redo json 
if args.gender_csv is not None: 
  args.gender_csv = args.gender_csv.split(',')
  # normal data, note: col names may not be same as our data 
  normal_fin_gender = pd.read_csv(args.gender_csv[1]) # name,0-2,3-9,10-19...
  # our data
  syndrome_fin_gender = pd.read_csv(args.gender_csv[0]) # powerpoint,slidenum,label,name,0-2,3-9,10-19...
  syndrome_fin_gender = syndrome_fin_gender.drop(columns='powerpoint,slidenum,label'.split(','))
  df_gender = pd.concat([syndrome_fin_gender,normal_fin_gender], ignore_index=True)
  df_gender = df_gender.drop_duplicates(subset = ["name"])
  df_gender = df_gender.sort_values(by=['name']).reset_index(drop=True)
  name = [re.sub ( 'detected_faces/','', i ) for i in df_gender['name'].values]
  name = [re.sub ( '_face0', '', i) for i in name ]
  gender_dict = dict(zip(name , df_gender['Female'].values)) # prob of female


# 
if not args.skip_normal: 
  df = pd.concat([syndrome_fin,normal_fin], ignore_index=True)
else: 
  df = syndrome_fin

df = df.drop_duplicates(subset = ["name"])
df = df.sort_values(by=['name']).reset_index(drop=True)
df.to_csv (args.new_df_path , index=False)

# ! make json 

json_label = {}
json_label['labels'] = []

label_type_dict = {'label1':args.disease, 'label2':args.age_bracket}

for idx, row in df.iterrows(): # name,path,label,fold,is_ext
  if args.skip_normal: 
    if "Normal" in row['name']: 
      continue
  onehot = make_label(row['label'],label_type_dict)
  # add gender ? why not. 
  if args.gender_csv is not None: # ! can we see some problem here ? 
    if row['name'] not in gender_dict: 
      print (row['path'])
      onehot = onehot + [ 0, 0 ] # no gender label? 
    else: 
      temp_ = np.round(gender_dict[row['name']],4)
      onehot = onehot + [ temp_ , np.round (1-temp_,4) ]
  #
  json_label['labels'].append ( [row['name'], onehot ] )


with open(args.json_path, 'w') as outfile:
  json.dump(json_label, outfile)


