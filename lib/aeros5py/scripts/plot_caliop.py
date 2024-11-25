#!/bin/bash

import xarray as xr
import matplotlib.pyplot as plt
import sys
import numpy as np
from pyhdf.HDF import HDF
import datetime
from pyhdf.VS import *

def read_CAL_L1(directory):
  data = xr.open_dataset(directory)
  Lidar_Data_Altitude = np.asarray(HDF(directory).vstart().attach('metadata')[0][29], dtype=float)
  data['Lidar_Data_Altitude'] = Lidar_Data_Altitude
  return data


d = read_CAL_L1(sys.argv[1])

plt.scatter(d.Longitude.data[25000:35000], d.Latitude.data[25000:35000 ])
plt.grid(which='both')


plt.show()



