
#!/home/farouk/anaconda3/bin/python
from os import  listdir,symlink, makedirs
from sys import argv, exit
import concurrent.futures
from math import sqrt
import numpy as np
from scipy.interpolate import RectBivariateSpline # interpolate isrf 2d
from glob import glob
input_path='/home/farouk/data/PRODUCT/'
output_path='./output/'
lat_min='20' #'15'
lat_max='20.5' #'25' #'49.22'
lon_min='130.5' #'-5'
lon_max='131' #'5'

workers=1

from netCDF4 import Dataset

def extract_nc_file(directory, list_all=None, CLOUDS=None, RADIANCE=None, IRRADIANCE=None , ISRF=None):
    filename = directory.split('/')[-1]
    if filename[13:15] == 'IR':  
        extract_irradiance_file(directory, list_all, IRRADIANCE)
    else :
      if filename[13:15] == 'RA': 
        if 'BAND' + filename[18:19] in RADIANCE.keys() : 
            extract_radiance_file(directory, list_all, RADIANCE)

      else :
        if filename[13:15] == 'CL': 
            extract_clouds_file(directory, list_all, CLOUDS)
        else :
          if filename[13:15] == 'SF' or filename[13:15] == 'IS':
              extract_isrf_file(directory, ISRF)

def extract_clouds_file(directory, list_all, CLOUDS):
    filename = directory.split('/')[-1]
    data = Dataset(directory, 'r')
    prefix = filename[13:15]
    print (filename )
    groups_str = ['DETAILED_RESULTS', 'GEOLOCATIONS', 'INPUT_DATA']

    for gr in groups_str :
      tmp_A = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups[gr]
      #print (str(gr) + ', ', end='' )
      for var in tmp_A.variables:
        if var in list_all : CLOUDS[var] = tmp_A.variables[var]
    for var in data.groups['PRODUCT'].variables:
      if var in list_all : CLOUDS[var] = data.groups['PRODUCT'].variables[var]


def main(input_path, output_path, lat_min, lat_max, lon_min, lon_max, workers):
  lat_min = float(lat_min)
  lat_max = float(lat_max)
  lon_min = float(lon_min)
  lon_max = float(lon_max)
  workers = int(workers)

  """#PREPARING DATA"""
  keys_list_common = ['satellite_altitude' , 'satellite_longitude', 'satellite_latitude']
  keys_list_clouds = ['latitude', 'longitude', 'cloud_fraction', 'qa_value', \
                     'cloud_top_pressure', 'cloud_top_pressure_precision', \
                     'cloud_fraction_precision', 'surface_albedo_fitted', 'surface_albedo_fitted_precision',\
                     'surface_altitude', 'surface_pressure', 'cloud_albedo_crb'] 

  list_all = keys_list_common + keys_list_clouds

  filenames = listdir(input_path) 
 

  CLOUDS = dict()

  [extract_nc_file(input_path+'/'+filename, list_all, CLOUDS) for filename in filenames ]


  """#FILTERING DATA

  ##Spatial domain
  """
  print('\nRESIZING ALONG SPATIAL DOMAINE : ')

  def mask_array(DCT, key) :
    DCT[key+'_domain'] = DCT[key][:].squeeze()[lat_start:lat_end, lon_start:lon_end]


  # CLOUD

  lat_list = (CLOUDS['latitude'][:].squeeze() > lat_min) * (CLOUDS['latitude'][:].squeeze() < lat_max)
  lon_list = (CLOUDS['longitude'][:].squeeze() > lon_min) * (CLOUDS['longitude'][:].squeeze() < lon_max)

  domain = np.argwhere(lon_list * lat_list)
  print(domain.shape)

  lat_start, lat_end = (min(domain[:,0]), max(domain[:,0]+1))
  lon_start, lon_end = (min(domain[:,1]), max(domain[:,1]+1))

  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor :
    r4 = [executor.submit(mask_array(CLOUDS, key)) for key in keys_list_clouds]

  print('CLOUDS...done', flush=True)

  """##Data quality filtering"""

  ## CLOUDS
  for var in keys_list_clouds :
    CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain'].data==CLOUDS['cloud_fraction_domain'].fill_value] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain'].data <0] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain'].data >1] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['qa_value_domain'].data < 0.5] = np.nan
  print ('CLOUDS...done', flush=True)

  print(CLOUDS.keys())
  print(CLOUDS['longitude_domain'][:].data.shape)


if __name__=='__main__' :
  main(input_path, output_path, lat_min, lat_max, lon_min, lon_max, 1)
