#!/bin/python

import sys, datetime
import numpy as np

param_file = sys.argv[1]
output_path = './'


param = np.loadtxt(param_file)


for p in param :
  pixel = p[0]
  utc0 = p[17]
  lat = p[2]
  lon = p[1]

  utc = datetime.datetime.fromtimestamp(float(utc0))

  with open(f'{output_path}/{int(pixel)}.UTtime', 'w') as f:
    f.write(utc.strftime('    %Y    %m    %d    %H    %M    %S'))
  with open(f'{output_path}/{int(pixel)}.latlong', 'w') as f:
    f.write(f'    {lat}     {lon}')
    
