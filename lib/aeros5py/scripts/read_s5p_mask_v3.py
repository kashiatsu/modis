#/home/farouk/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Extracts a delimited region from S5P radiance (.nc) files

args:
    input_path (str): Must contain RADIANCE, IRRADIANCE, and CLOUDS files.
    output_path (str)
    lat_min (str)
    lat_max (str)
    lon_min (str)
    lon_max (str)
    cloud_fraction_max (str): Not used.
    isrf_file (str): The unsampled version.
    workers (int): not implemented (use 1)
    
    Note:
        The difference between lon_max and lon_min must be > 0.1 degree.
        This script doesn't support ISRF extraction.

    10/2020 batch mode works
"""

__author__ = "Farouk Lemmouchi"
__version__ = "3.0.0" # batch mode works
__email__ = "farouk.lemmouchi@lisa.u-pec.fr"


# Warinings : overflow while converting Irradiance --> dtype=np.float64
from os import  listdir,symlink, makedirs, path 
from sys import argv, exit
import concurrent.futures
from math import sqrt
import numpy as np
from scipy.interpolate import RectBivariateSpline # interpolate isrf 2d
from glob import glob
import gc, dill
from aeros5py.extract_1d import *
import time
#hidden imports
import pkg_resources.py2_warn, cftime



batch_mode=True
do_isrf=False
do_write=True
do_err=True
do_l1b=True
do_target=True
#writing_factor=13
writing_factor=1
do_viirs_mask=False

batch_size = 70000 #max files per extract
print( 'Version = ', __version__ ) 
print( 'writing_factor = ', writing_factor )
print( 'batch_mode = ', batch_mode ) 

# std_band of which swath resolution will be used, if changed might have to change midswath!
std_band = 'BAND3' 

def main(input_path, output_path, lat_min, lat_max, lon_min, lon_max, cloud_fraction_max, isrf_file, workers):
  print (locals())

  lat_min = float(lat_min)
  lat_max = float(lat_max)
  lon_min = float(lon_min)
  lon_max = float(lon_max)
  workers = int(workers)

  cloud_fraction_max = float(cloud_fraction_max)
  """#PREPARING DATA"""
  keys_list_radiance = ['scanline', 'ground_pixel', 'latitude', 'longitude' , 'radiance', 'quality_level', 'radiance_error', 'ground_pixel_quality']
  keys_list_angles = ['longitude_bounds' , 'latitude_bounds', 'viewing_zenith_angle', \
                       'solar_zenith_angle', 'solar_azimuth_angle','viewing_azimuth_angle']
  keys_list_common = ['satellite_altitude' , 'satellite_longitude', 'satellite_latitude'] #no ground pixel variables
  keys_list_clouds = ['cloud_fraction', 'qa_value', \
                     'cloud_top_pressure', 'cloud_top_pressure_precision', \
                     'cloud_fraction_precision', 'surface_albedo_fitted', 'surface_albedo_fitted_precision',\
                     'surface_altitude', 'surface_pressure', 'cloud_albedo_crb'] 
                     #'latitude', 'longitude', \
  keys_list_ai = ['aerosol_index_340_380'] #aerosol_index_354_388']

  keys_list_gpixel = ['nominal_wavelength', 'irradiance', 'irradiance_error'] #no scanline variables
  
  list_all = keys_list_radiance + keys_list_angles + keys_list_common + keys_list_clouds + keys_list_gpixel + ['delta_time']

  #isrf_path = '/home/farouk/data/isrf_release/isrf/binned_uvn_swir_sampled/' #(27 lambdas)

  filenames = listdir(input_path) 
  print(filenames)
 
  #filenames=['S5P_OFFL_L2__CLOUD__20191222T040256_20191222T054426_11347_01_010107_20191223T174903.nc','S5P_OFFL_L1B_RA_BD3_20191222T040256_20191222T054426_11347_01_010000_20191222T072813.nc']


  CLOUDS = dict()
  UVAI = dict() #UV Aerosol index
  ISRF =  {'BAND3':dict(), 'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}
  RADIANCE = {'BAND3':dict(), 'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}
  #RADIANCE = {'BAND3':dict(),'BAND5':dict() }
  IRRADIANCE = {'BAND3':dict() ,'BAND4':dict(),'BAND5':dict() ,'BAND6':dict()}
  #IRRADIANCE = {'BAND3':dict() ,'BAND5':dict() }

  [extract_nc_file(input_path+'/'+filename, list_all, CLOUDS, UVAI, RADIANCE, IRRADIANCE , ISRF) for filename in filenames ]

  if do_isrf :
    extract_nc_file(isrf_file, ISRF=ISRF)

  print ('\nBANDS LIMITS : ')
  for bd in RADIANCE.keys() :
    print (bd, end=' ')
    print(RADIANCE[bd]['nominal_wavelength'][0,0,0], RADIANCE[bd]['nominal_wavelength'][0,0,-1])

  """#FILTERING DATA

  ##Spatial domain
  """
  print('\nRESIZING ALONG SPATIAL DOMAINE : ')
  STD = RADIANCE[std_band]
  std_shape = STD['latitude'][:].squeeze().shape
  print(f'\nRegular grid : {std_band}(time, scanline, ground_pixel) = {std_shape}\n')


  def make_2d_mask(DCT, lonlim, latlim) : 
    mask450 = ( (DCT['longitude'][:].squeeze() < lonlim[1]) & (DCT['longitude'][:].squeeze() > lonlim[0]) & 
              (DCT['latitude'][:].squeeze() < latlim[1]) & (DCT['latitude'][:].squeeze() > latlim[0]) )
    mask448 = mask450[:,:-2]

    return {'448':mask448, '450':mask450 }

  #def mask_array2(DCT, key,lonlim, latlim ):
  #   DCT[key][DCT['longitude²V 

  def mask_array(DCT, key, mask=None) :
    if mask is not None :
      try :
        DCT[key+'_domain'] = DCT[key][:].squeeze()[mask['448']].data
      except :
        try :
          DCT[key+'_domain'] = DCT[key][:].squeeze()[mask['450']].data
        except :
          print('here')
    else : #no mask
      DCT[key+'_domain'] = DCT[key][:].squeeze().data[lat_start:lat_end, lon_start:lon_end]



  for bd in RADIANCE :
    RADIANCE[bd]['ground_pixel'], RADIANCE[bd]['scanline'] = np.meshgrid(RADIANCE[bd]['ground_pixel'][:], RADIANCE[bd]['scanline'][:] )

 
  # apply masking
  coo_mask = make_2d_mask(STD, [lon_min, lon_max], [lat_min, lat_max] )
  
  # mask timedelta
  STD['deltat'] = np.repeat( STD["delta_time"][:], std_shape[1] ).reshape( (1,*std_shape) ) 
  mask_array(STD, 'deltat' , mask=coo_mask)


  # expand dimension of the other variables to apply the same mask
  for var in keys_list_common :
    #STD[var+'_domain'] = np.empty(std_shape) # initialize
    STD[var] = STD[var][:].repeat(std_shape[1]).reshape(std_shape)
    mask_array(STD, var, mask=coo_mask)
  

  for bd in RADIANCE.keys():
    print (f'{bd}...', end='', flush=True)
    if write_l1b :
      IRRADIANCE[bd]['nominal_wavelength_domain'] = IRRADIANCE[bd]['nominal_wavelength'][:].squeeze().squeeze().data#[lon_start:lon_end, :]
      IRRADIANCE[bd]['irradiance_domain'] = IRRADIANCE[bd]['irradiance'][:].squeeze().squeeze().data#[lon_start:lon_end, :]
      IRRADIANCE[bd]['irradiance_error_domain'] = IRRADIANCE[bd]['irradiance_error'][:].squeeze().squeeze().data#[lon_start:lon_end, :]
    RADIANCE[bd]['nominal_wavelength_domain'] = RADIANCE[bd]['nominal_wavelength'][:].squeeze().squeeze().data#[lon_start:lon_end, :]

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor :
      r3 = [executor.submit(mask_array(RADIANCE[bd], key, mask=coo_mask)) for key in keys_list_radiance+keys_list_angles]

    #Exit if beyond domain
    a = STD['longitude_domain'] < lon_min
    b = STD['longitude_domain'] > lon_max
    c = a*b
    d = np.where(c == False) 
    if d[0].size < 500 : #extract the orbit pass only if it contains more than 1000 pixels
      print('Avaliable pixels on orbit : ', d[0].size)
      print('\nSkiped : orbit beyond domain too')
      exit()

    #if all(STD['longitude_domain'] < lon_min) or all(STD['longitude_domain'] > lon_max) :

    # interpolate isrf to fit radiance wavelength sampling
    if do_isrf :
      print('interpolating ISRF...', end='', flush=True)
      ISRF[bd]['isrf_domain_resampled0'] = np.zeros( [RADIANCE[bd]['radiance_domain'].shape[1] , RADIANCE[bd]['radiance_domain'].shape[2], 257] )+999999
      
      for idx_lon in range(RADIANCE[bd]['radiance_domain'].shape[1]) :
        a = RectBivariateSpline(ISRF[bd]['wavelength_domain'][:][idx_lon,:], ISRF[bd]['delta_wavelength'][:], ISRF[bd]['isrf_domain'][idx_lon,:,:])
        
        for idx_wv, wv in enumerate(RADIANCE[bd]['nominal_wavelength_domain'][idx_lon,:].data) :
          for idx_dlambda, dlambda in enumerate(ISRF[bd]['delta_wavelength'][:].data) :
            ISRF[bd]['isrf_domain_resampled0'][idx_lon,idx_wv,idx_dlambda] = a(wv,dlambda)
      print ('done', flush=True)

  # CLOUD
  #lat_list = (CLOUDS['latitude'][:].squeeze() > lat_min) * (CLOUDS['latitude'][:].squeeze() < lat_max)
  #lon_list = (CLOUDS['longitude'][:].squeeze() > lon_min) * (CLOUDS['longitude'][:].squeeze() < lon_max)
  #domain = np.argwhere(lon_list * lat_list)
  #lat_start, lat_end = (min(domain[:,0]), max(domain[:,0]+1))
  #lon_start, lon_end = (min(domain[:,1]), max(domain[:,1]+1))
  
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor :
    r4 = [executor.submit(mask_array(CLOUDS, key, mask=coo_mask)) for key in keys_list_clouds]
  print('CLOUDS...done', flush=True)

  #UVAI new
  #with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor :
  #  r4 = [executor.submit(mask_array(UVAI, key, mask=coo_mask)) for key in keys_list_ai]
  #print('UVAI...done', flush=True)

  domain = STD['ground_pixel_domain']
  if batch_mode : total_pixels = coo_mask['450'].sum()
  if not batch_mode : total_pixels = coo_mask['450'].sum()/writing_factor
  
  print (f"\nTOTAL PIXELS NUMBER {total_pixels}", flush=True)
  if batch_mode == False : 
    if total_pixels > batch_size : 
      print('Exit : Domain too large' )
      exit()


  irregular = []
  for bd in RADIANCE.keys():
      if RADIANCE[bd]['ground_pixel_domain'].size != domain.size :
          irregular.append(bd)
  print (f"irregular grids : {irregular}")

  """##Data quality filtering"""

  print ('\nDATA QUALITY FILTERING :', flush=True)

  ## RADIANCE
  #radiance_noise radiance_error quality_level spectral_channel_quality ground_pixel_quality
  def filter_radiance(bd) :
    RADIANCE[bd]['radiance_domain'][RADIANCE[bd]['radiance_domain'] < 0] = 0
    RADIANCE[bd]['radiance_domain'][RADIANCE[bd]['quality_level_domain'] < 2] = RADIANCE[bd]['radiance'].fill_value
    RADIANCE[bd]['ground_pixel_quality_domain'][RADIANCE[bd]['ground_pixel_quality_domain'] >= 4] = RADIANCE[bd]['radiance'].fill_value
    print(f'{bd}...done', flush=True)

  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor :
    e = [executor.map(filter_radiance, RADIANCE.keys())]
    print('RADIANCE...done')
  if do_isrf :
    # isrf bad values
    for bd in ISRF.keys():
      np.where(ISRF[bd]['isrf_domain_resampled0'] < 0, 0,ISRF[bd]['isrf_domain_resampled0'])

    #for i in range(rad_shape[0]):
    #    if CLOUDS['cloud_fraction'][0,i,j] > cloud_fraction_max :
    #      RADIANCE[bd]['radiance'][0,i,j,:].fill(RADIANCE[bd]['radiance'].fill_value)

  #RADIANCE[bd]['radiance'] = RADIANCE[bd]['radiance'].filled(fill_value=np.nan)
  ## CLOUDS
  for var in keys_list_clouds :
    #CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain']==CLOUDS['cloud_fraction'].fill_value] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain'] <0] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['cloud_fraction_domain'] >1] = np.nan
    CLOUDS[f'{var}_domain'][CLOUDS['qa_value_domain'] < 0.5] = np.nan
  print ('CLOUDS...done', flush=True)

  #any(RADIANCE[bd]['spectral_channel_quality'][0,i,j,:])
  #RADIANCE[bd]['radiance'][:][rad_spectral_channel_quality[:] != 0] = RADIANCE[bd]['radiance']._FillValue

  """#UPSAMPLING"""

  print('\nUPSAMPLING :\n', flush=True)
  special_shape = ['latitude_bounds_domain' , 'longitude_bounds_domain',  \
                   'radiance_domain', 'radiance_error_domain','quality_level_domain']

  # colocalizing HR with LR pixels
  def no_compute_nearest1d(LR_DICT, HR_DICT): #WARNINIG : less precise use compute_nearest_px_to_HR_VEC2
    """Adds key to lower resolution dictionary: idx
    represent the index of the closest pixel in the heigh resolution grid"""
    
    LR_DICT['idx'] = np.zeros(HR_DICT['longitude_domain'].size)*-9999
    for i,HR_lon in enumerate(HR_DICT['longitude_domain']) :
      HR_lat = HR_DICT['latitude_domain'][i]
      d = (LR_DICT['longitude_domain']-HR_lon)**2 + (LR_DICT['latitude_domain']-HR_lat)**2
      LR_DICT['idx'][i] = np.nanargmin(d)
      
  def upsample1d(LR_DICT, idx, var) :
    #print(var)
    #print(LR_DICT[var].shape)
    #print(idx.shape)

    if var in special_shape :
      LR_DICT[var+'_resampled'] = np.zeros( [idx.size, LR_DICT[var].shape[-1] ])*np.nan
     
      for i in range(idx.size) :
        if idx[i] == -9999 : pass 
        else : LR_DICT[var+'_resampled'][i,:] = LR_DICT[var][int(idx[i]),:]
    else :
      LR_DICT[var+'_resampled'] =  np.zeros(idx.size)*np.nan
      for i in range(idx.size) :
        if idx[i] == -9999 : pass 
        else : LR_DICT[var+'_resampled'][i] = LR_DICT[var][ int(idx[i]) ]
    

  # code start
  # upsample radiances and isrf
  for bd in RADIANCE.keys():
    print(f"{bd}...", end='', flush=True)

    if bd not in irregular : continue # case treated while writing
    else :
      no_compute_nearest1d(RADIANCE[bd], STD) #quicker

      for var in ['radiance', 'radiance_error'] : # keys_list_radiance :
        upsample1d(RADIANCE[bd], RADIANCE[bd]['idx'], var+'_domain')
    print('done', flush=True)

  # upsample CLOUDS
  print('CLOUDS...', end='', flush=True)
  if CLOUDS['cloud_fraction_domain'].shape == domain.shape : 
    for var in keys_list_clouds :
      CLOUDS[var+'_domain_resampled'] = CLOUDS[var+'_domain']
  else :
    no_compute_nearest1d(CLOUDS, STD) #quicker
    for var in keys_list_clouds :
      upsample1d(CLOUDS, CLOUDS['idx'], var+'_domain')
  print('done', flush=True)

  """#CONVERSION

  ##angles
  """

  STD['VAA_vlidort'] = -STD['viewing_azimuth_angle_domain']+180

  STD['SAA_vlidort'] = -STD['solar_azimuth_angle_domain']

  #STD['RELAZM_vlidort'] = STD['SAA_vlidort'] - ( STD['VAA_vlidort'] - 180.)

  STD['RELAZM_vlidort'] = STD['viewing_azimuth_angle_domain']-STD['solar_azimuth_angle_domain'] -180

  """##Rad/Irrad errors"""

  #Conversion from decibel to mol.m-2.nm-1.sr-1.s-1
  if write_l1b :
    for bd in RADIANCE.keys():
      tmp = RADIANCE[bd]['radiance_error_domain']/10
      tmp2 = 10**(-tmp)
      RADIANCE[bd]['radiance_error_domain'] = RADIANCE[bd]['radiance_domain']*tmp2
      tmp = IRRADIANCE[bd]['irradiance_error_domain']/10
      tmp2 = 10**(-tmp)
      IRRADIANCE[bd]['irradiance_error_domain'] = IRRADIANCE[bd]['irradiance_domain']*tmp2


  """##CLEANING MEMORY"""
  for bd in RADIANCE.keys():
    for var in list(RADIANCE[bd]):
      if var in list_all : del RADIANCE[bd][var]
    try :
      for var2 in ['isrf', 'wavelength', 'fwhm'] :
        del ISRF[bd][var2]
    except : pass 
  try :
    for var in list(CLOUDS) :
      if var in list_all : del CLOUDS[var]
  except : pass
  try :
    for var in ['latitude_domain','longitude_domain']:
      del CLOUDS[var]
  except : pass

  gc.collect()

  """#WRITING DATA"""


  if do_write :
    for i in filenames : # to catch the date
      if i[13:15]=='RA':  
          date = i
    if do_viirs_mask : #new
      from aeros5py import extract_viirs
      vlon, vlat, vcod = extract_viirs(date, lat1=lat_min,  lat2=lat_max , lon1=lon_min , lon2=lon_max, threshold=3, output_path='', var='Integer_Cloud_Mask')


    date1 = datetime.datetime.strptime(date[20:28], '%Y%m%d')
    #utc_time = date1 + datetime.timedelta(milliseconds=int(STD["delta_time_domain"][0])) #WARNING : THIS IS THE FIRST DOMAIN PIXEL

    #pixel_flags = {0:'no_error', 1: 'solar_eclipse', 2: 'sun_glint_possible', 4: 'descending', 8: 'night', 16: 'geo_boundary_crossing', 128: 'geolocation_error'}

    # start code
    print('\nWRITING FILES : ', flush=True)
    output_path2 = None
    #output_path2 = './output2/' # uncomment this to use a second drive.
    increment = None
    output_path_root = output_path
    output_path_root2 = output_path2
    if total_pixels > batch_size  and batch_mode :
      increment = batch_size # with respect to longitude
      print (f'\nincrement = {increment}')
      nbatch = (total_pixels // batch_size)+1
      print(f'\nThere will be {nbatch} extractions.\n')
      for n in range(nbatch) : 
        output_path = f'{output_path_root}/{str(n)}' 
        create_output_directories(output_path, do_write, do_isrf, do_l1b, do_err, nbatch)
        if output_path2 is not None : 
          output_path2 = f'{output_path_root2}/{str(n)}'
          create_output_directories(output_path2, do_write, do_isrf, do_l1b, write_err, nbatch)
    else :
      create_output_directories(output_path_root, do_write, do_isrf, do_l1b, write_err)
    
    #n_existing = len(glob(output_path+'/err/*')) # append pixels (multiple passes)
    #pixnum = 10000 #+ n_existing # append pixels (multiple passes) 
    #if n_existing > (5/6*batch_size) :
    #  print('Orbit already has too many pixles')
    #  exit()

    start = time.time()

    #if increment is None :
    #  list_param = open(f"{output_path}/target/list.s5p_param", 'a') 
    #  list_cf = open(f"{output_path}/target/list.cloudfree", 'a')

    for i, j in enumerate( STD['ground_pixel_domain'] ) :
        n = (i // increment)
        output_path=f"{output_path_root}/{str(n)}/"
        if path.isfile(f"{output_path}/target/list.s5p_param") == False : 
          pixnum = 10000
          list_param = open(f"{output_path}/target/list.s5p_param", 'a') 
          list_cf = open(f"{output_path}/target/list.cloudfree", 'a')
          #if UVAI['aerosol_index_340_380_domain'][i] > 0 :
          #if j % writing_factor == 0 : # (condition to reduce pixel number)
          utc_time = date1 + datetime.timedelta(milliseconds=int(STD['deltat_domain'][i]))
        pixnum = pixnum + 1
        #output_path2 = f"{output_path_root2}/{str(n)}/" if output_path2 is not None else output_path

        list_cf.write(f"{STD['longitude_domain'].data[i]}\t{STD['latitude_domain'].data[i]}\t0.0\tS5P_L1B_4ch_{date[20:28]}_{str(pixnum)}.dat\n")
        append_param(list_param, STD, CLOUDS, i, j, pixnum, utc_time) 
        pixnum = write_pixel(i, j, STD, RADIANCE, IRRADIANCE, CLOUDS, ISRF, 
                                    do_isrf, do_l1b, do_err, do_target, 
                                    output_path, utc_time, 
                                    irregular, increment=increment, output_path2=output_path2, pixnum=pixnum)

    list_param.close()
    list_cf.close()
    end = time.time()

    print(f"{total_pixels} pixels written in {end-start} seconds ({output_path_root2}).")

  if increment is None:
    print('Finished')
  else:
    print('Finished : batch mode')

  # Get domain limits
  for b in  listdir(output_path_root) : 
    if path.isdir(f'{output_path_root}/{b}') : 
      coo = np.genfromtxt(f'{output_path_root}/{b}/target/list.cloudfree') 
      with open(f'{output_path_root}/{b}/domain', 'w') as dom : 
        dom.write('lon1='+str(np.round(coo[:,0].min()-0.05, decimals=2))+'; ') 
        dom.write('lon2='+str(np.round(coo[:,0].max()+0.05, decimals=2))+'; ') 
        dom.write('lat1='+str(np.round(coo[:,1].min()-0.05, decimals=2))+'; ') 
        dom.write('lat2='+str(np.round(coo[:,1].max()+0.05, decimals=2))+'; ') 

  ##Extract VIIRS 
  #print ( 'Extracting viirs...')
  #from aeroviirs import extract_viirs
  #extract_viirs.main(date=date, output_path='/DATA/SPECAT_2/farouk/VIIRS/' + date, lon1=lon_min, lon1=lon_max, lat1=lat_min, lat2=lat_max )
    
if __name__=='__main__' :
  main(*argv[1:])
