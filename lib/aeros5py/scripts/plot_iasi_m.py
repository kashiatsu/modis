#!/bin/python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import sys

#date=sys.argv[1]
#dates =['20191218_iasi_issue', '20191223', '20191224', '20191222']
dates=['20191224']#, '20191225_iasi']
fig = plt.figure(figsize=(17,15))
for i,date in enumerate(dates) :
  #file=f'/DATA/SPECAT_2/farouk/AUSTRALIA_BIG3/INPUTS/{date}/IASI_spectra/list.cloudfree'
  #file=f'/DATA/SPECAT_2/farouk/AUSTRALIA_BIG3/INPUTS/{date}/Atm_Surface_Target_UV_TIR/list.cloudfree_iasipixels'
  #file=f'/DATA/SPECAT_2/farouk/AUSTRALIA_BIG3/INPUTS/{date}/Atm_Surface_Target_UV_TIR/list.cloudfree'
  file=f'/DATA/OTHERS_2/SPECAT/farouk/tmp/target/list.cloudfree'
  #file=f'/home/flemmouchi/pre_proc/compile/output/target/list.cloudfree'
  print(file)
  ax=plt.subplot(1,1,i+1,projection=ccrs.PlateCarree())
  data = pd.read_csv(file, delim_whitespace=True)
  plt.scatter(data.iloc[:,0], data.iloc[:,1], 3)
  plt.title('IASI_grid '+date)
  ax.coastlines()
plt.show()
#plt.savefig('IASI_grid_'+date)
