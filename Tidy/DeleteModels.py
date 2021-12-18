
import os,sys,re,pickle 
import json 
import numpy as np 

rootpath = '/data/duongdb/WS22qOther_12082021/Stylegan3Model/TfStyleGAN3LabelRes256SkipNormal+Gender'

# ! look at best 

for f in os.listdir(rootpath): 
  metric = open(os.path.join(rootpath,f,'metric-fid50k_full.jsonl'),'r')
  metric_array = []
  name = []
  for l in metric: 
    z = json.loads(l)
    metric_array.append ( z['results']['fid50k_full'] ) 
    name.append ( z["snapshot_pkl"] )
  low_point = np.min ( np.array(metric_array) ) 
  metric.close()
  print (f, name[metric_array.index(low_point)], low_point)
  


# ! delete 

for f in os.listdir(rootpath): 
  # if 'stylegan3-t' not in f: 
  #   continue
  metric = open(os.path.join(rootpath,f,'metric-fid50k_full.jsonl'),'r')
  metric_array = []
  for l in metric: 
    z = json.loads(l)
    metric_array.append ( z['results']['fid50k_full'] ) 
  low_point = np.sort ( np.array(metric_array) ) [4] # ! 2nd last 
  metric.close()
  # read again 
  metric = open(os.path.join(rootpath,f,'metric-fid50k_full.jsonl'),'r')
  for l in metric: 
    z = json.loads(l)
    if z['results']['fid50k_full'] > low_point: 
      if os.path.exists(os.path.join(os.path.join(rootpath,f,z['snapshot_pkl']))): 
        os.system('rm ' + os.path.join(os.path.join(rootpath,f,z['snapshot_pkl'])) )
        temp_ = re.sub('network-snapshot-','fakes',z['snapshot_pkl'])
        temp_ = re.sub('pkl','png',temp_)
        os.system('rm ' + os.path.join(os.path.join(rootpath,f,temp_)) ) # network-snapshot-001000.pkl --> fakes001000.png
  #
  metric.close()
  os.system ( 'rm ' + os.path.join(rootpath,f,'fakes_init.png') )
  # os.system ( 'rm ' + os.path.join(rootpath,f,'reals.png') )

#

