import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


coloc = np.loadtxt('/DATA/SPECAT_2/farouk/AEROS5P_MAX/codes_source/scripts/list_pixels_20191222.txt', dtype=int)
iasi = pd.read_csv('/DATA/SPECAT_2/farouk/AEROS5P_MAX/codes_source/scripts/list.cloudfree', sep='\t')
found = [] 
for j, i in enumerate(iasi.iloc[:,3]) : 
    if int(i[21:26]) in coloc : 
         found.append((iasi.iloc[j, 0], iasi.iloc[j,1]))
a = np.asarray(found[:])
ax = plt.subplot(1,1,1, projection=ccrs.PlateCarree())
plt.scatter(a[:,0], a[:,1], s=0.3)
ax.coastlines()
plt.grid()
plt.savefig('verify_viirs.png')
