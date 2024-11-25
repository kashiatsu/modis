#/bin/python

import numpy as np
import struct
date = "20191222"
path = '/DATA/SPECAT_2/farouk/VIIRS/' 
output_path = path + date
#output_path="./"
with open(f'{output_path}/list.viirs_cloudfree.bin', 'rb') as f :
  n = np.int.from_bytes(f.read(4), byteorder="little")
  print(n)
  longitude = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=0) 
  latitude = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=4) 
  cloud_od = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=4) 

print (cloud_od)
quit()
#PLOT
import cartopy.crs as ccrs 
from matplotlib.axes import Axes 
from cartopy.mpl.geoaxes import GeoAxes 
GeoAxes._pcolormesh_patched = Axes.pcolormesh 
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(17,17), constrained_layout=True)
ax = plt.subplot(1,1,1,projection=ccrs.PlateCarree())

longitude = np.where(cloud_od <3, np.nan, longitude)
latitude = np.where(cloud_od <3,np.nan, latitude)

im = plt.scatter(longitude, latitude) 
plt.xlim([110, 150])
plt.ylim([-40, -10])
plt.title('VIIRS cloud optical depth')
ax.coastlines() 
print("saveing")
plt.savefig('frombin.png', bbox_inches = 'tight', pad_inches = 0.2)
