#!/home/farouk/anaconda3/bin/python3.7
from os import  listdir
from sys import  exit
#import cartopy.crs as ccrs
from aeros5py.extract import *
from netCDF4 import Dataset
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

input_path='/DATA/SPECAT/farouk/S5P_L1B/PRODUCT_IDF/'
#isrf_file='/home/farouk/data/isrf_release/isrf/binned_uvn_spectral_unsampled/S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc'

CLOUDS = dict()
ISRF =  {'BAND3':dict(), 'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}
RADIANCE = {'BAND3':dict(), 'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}
IRRADIANCE = {'BAND3':dict() ,'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}

keys_list_radiance = ['latitude', 'longitude' , 'radiance', 'quality_level', 'radiance_error']
keys_list_angles = ['longitude_bounds' , 'latitude_bounds', 'viewing_zenith_angle', \
                   'solar_zenith_angle', 'solar_azimuth_angle','viewing_azimuth_angle']
keys_list_common = ['satellite_altitude' , 'satellite_longitude', 'satellite_latitude']
keys_list_clouds = ['latitude', 'longitude', 'cloud_fraction', 'qa_value', \
                 'cloud_top_pressure', 'cloud_top_pressure_precision', \
                 'cloud_fraction_precision', 'surface_albedo_fitted', 'surface_albedo_fitted_precision',\
                 'surface_altitude', 'surface_pressure', 'cloud_albedo_crb'] 

list_all = keys_list_radiance + keys_list_angles + keys_list_common + keys_list_clouds + ['nominal_wavelength', 'irradiance', 'irradiance_error']

filenames = listdir(input_path)
[extract_nc_file(input_path+'/'+filename, list_all, CLOUDS, RADIANCE, IRRADIANCE , ISRF) for filename in filenames ]

#extract_nc_file(isrf_file, ISRF=ISRF)

#----------PLOTS-----------#

#plot_spectra(IRRADIANCE, var='irradiance', xmin=333, xmax=444)
#plot_spectra(RADIANCE,  groundpixel=50, scanline=43)
#plot_spectra(RADIANCE, scanline=6,  DICT2=IRRADIANCE)
#plot_spectra(IRRADIANCE,  groundpixel=50,var='irradiance', scale='log')
#plot_isrf(ISRF, 45, iwavelength=65)




bd = 'BAND3'
for i, var in enumerate([ 'cloud_fraction'] ) : #'viewing_azimuth_angle', 'solar_azimuth_angle', 'solar_zenith_angle', 'viewing_zenith_angle', ]) :
  ax = plt.subplot(1,1,i+1, projection=ccrs.PlateCarree())
  #plt.pcolormesh(RADIANCE[bd]['longitude'][0,:,:], RADIANCE[bd]['latitude'][0,:,:], RADIANCE[bd][var][0,:,:],cmap=plt.cm.jet)
  plt.pcolormesh(CLOUDS['longitude'][0,:,:], CLOUDS['latitude'][0,:,:], CLOUDS[var][0,:,:],cmap=plt.cm.jet, vmin=0, vmax=0.1)
  ax.coastlines()
  plt.title(f's5p_{var}')
  plt.grid()
  plt.xlabel('longitude')
  plt.ylabel('latitude')
  ax.set_xlim( xmin=-19, xmax=55)
  ax.set_ylim( ymin=-43, ymax=50)
  plt.colormaps()
  plt.colorbar()
plt.savefig(f'scatter_SA_20190910_{var}_s5p.jpeg')
#plt.show()

