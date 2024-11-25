import matplotlib.pyplot as plt
import numpy as np
#import sys
#import pandas as pd
import cartopy.crs as ccrs
#import netCDF4 as nc
import xarray as xr
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

filename1 ='/DATA/SPECAT_2/farouk/MODIS/20191223/g4.timeAvgMap.MOD08_D3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean.20191223-20191223.125E_40S_155E_10S.nc' 
data = xr.open_dataset(filename1) 
#var = 'Aerosol_Layer_Fraction'
#var = 'Extinction_Coefficient_532'
#var = 'Relative_Humidity'
#var = 'Temperature'
#data = nc.Dataset(filename1, 'r')

var= 'AOD 550 nm'
fig = plt.Figure()
projection=ccrs.PlateCarree()
ax=plt.axes(projection=projection)
plt.pcolormesh(data.lon.data, data.lat.data,data.MOD08_D3_6_1_Deep_Blue_Aerosol_Optical_Depth_550_Land_Mean.data, cmap=plt.cm.jet,vmin=0, vmax=1)
#ax.set_extent([data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()])
lat1=-40.
lat2=-10.
lon1=125.
lon2=155.
step=5
plt.colorbar()
plt.xticks(np.arange(lon1,lon2, step=step));
plt.yticks(np.arange(lat1,lat2, step=step));
ax.set_xlim( xmin=lon1, xmax=lon2)
ax.set_ylim( ymin=lat1, ymax=lat2)
ax.grid()
#ax.set_xticklabels(data.lon.data)
#ax.set_yticklabels(data.lat.data)
ax.coastlines()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('AOD 550 nm')
plt.savefig('tmp.png')
