#!/bin/python3.6
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import sys

directory = sys.argv[1]
file = directory.split('/')[-1]
print(file)
ax=plt.subplot(1,1,1,projection=ccrs.PlateCarree())
data = pd.read_csv(directory, delim_whitespace=True)
plt.scatter(data.iloc[:,0], data.iloc[:,1], 3)
plt.title(file)
ax.coastlines()

plt.savefig(file + '.png')
