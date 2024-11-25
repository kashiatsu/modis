#!/bin/python

from os import  listdir
from sys import  exit
import numpy as np
import matplotlib.pyplot as plt
from aeros5py import *

ouput_path = './output'
isrf_file='/home/farouk/data/isrf_release/isrf/binned_uvn_spectral_unsampled/S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc'
#isrf_file='/home/farouk/data/isrf_release/isrf/binned_uvn_swir_sampled/S5P_OPER_AUX_ISRF___00000101T000000_99991231T235959_20180320T084215.nc'

ISRF =  {'BAND3':dict(), 'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}

extract_nc_file(isrf_file, ISRF=ISRF)

[ np.where( ISRF[bd]['isrf'][:].data < 0, 0, ISRF[bd]['isrf'][:].data ) for bd in ISRF.keys()]
print (ISRF['BAND3']['ground_pixel'].shape)
exit()
for j in ISRF['BAND3']['ground_pixel'].shape :
    swath = 10000 + j
    with open(f'{output_path}/isrf/s5p_isrf_{swath}.bin', 'wb' ) as isrf_file : # ISRF file
        for bd in ISRF.keys():
          np.float64(ISRF[bd]['wavelength'][:]).tofile(isrf_file)
          np.float64(ISRF[bd]['isrf'][:][j,:,:]).tofile(isrf_file)
          np.float64(ISRF[bd]['delta_wavelength'][:].data).tofile(isrf_file)
