#!/bin/python
from aeroviirs import extract_viirs
from aeros5py.utils import *
from aeros5py.post_proc import read_CAL_L1
import pickle
from sys import exit
from glob import glob
import numpy as np
import pandas as pd


date = '20191222'


CAL_name = glob(f'/DATA/SPECAT_2/farouk/lidar/EARTHDATA/L1/D/CAL_*{date[0:4]}-{date[4:6]}-{date[6:8]}*.hdf')
print(CAL_name)
CAL = read_CAL_L1(CAL_name[1])
mask = make_1d_mask(CAL, [139.17, 146.34], [-39.0, -13.1])
if CAL.Subsatellite_Latitude.data[mask][0].size == 0 : print("ERROR: LIDAR doesn't cross domain"); exit()

clon = CAL.Longitude.data[mask]
clat = CAL.Latitude.data[mask]
print(clon.shape)

param = np.loadtxt('list.s5p_param')
param = pd.DataFrame(param)
list = np.loadtxt('list_pixels_20191222.txt')

notfound = []
for idx , i in enumerate(param.iloc[:,0]) :
  if i not in list :
    notfound.append(idx)

print(param.shape)
param.drop(notfound, inplace=True)
param.reset_index(inplace=True, drop=True)




#longitude, latitude, cod = extract_viirs.main(date, threshold=3, output_path='', var='Integer_Cloud_Mask', lat1=-39.0, lat2=-13.1, lon1=139.17, lon2=146.34)
#print(longitude.shape)

idx = colocate_generic(param.iloc[:,1], param.iloc[:,2], clon, clat, lonlim=[139.17, 146.34], latlim=[-39.0, -13.1])
print(idx)

#viirs_out = np.empty([idx.size,3])
#for n,i in enumerate(idx) :
#  viirs_out[n,:] = longitude[i], latitude[i], cod[i]
#
#with open('viirs_out', 'wb') as f :
#  f.write(pickle.dumps(viirs_out))

out = np.empty([idx.size,3])
for n,i in enumerate(idx) :
  #print(  str(param.iloc[i,0:3]) + '\n')
  out[n,:] = param.iloc[i,0:3]

with open('cal_out', 'wb') as f :
  f.write(pickle.dumps(out))


#with open('trans_coo', 'rb') as f:
#  trans_coo = pickle.load(f)
#idx = colocate_generic(trans_coo[:,1], trans_coo[:,2], longitude, latitude, lonlim=(139.17,146.34), latlim=(-38.0,-13.1))
#for i in idx :
#  print(trans_coo[i,0])




