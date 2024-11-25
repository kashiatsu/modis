#!/home/flemmouchi/.conda/envs/py36/bin/python3.6

from netCDF4 import Dataset
import datetime
import os


def create_output_directories(output_path, do_write, do_isrf, write_l1b, write_err, nbatch=None):
  """Creates output directory tree"""
  from os import makedirs
  if do_isrf :
    try : 
      if nbatch is not None : 
        makedirs(f'{output_path}/../isrf/')
      else :
        makedirs(f"{output_path}/isrf/")
    except : pass
  if write_l1b :
    try :
      makedirs(f"{output_path}/l1b/")
    except : pass
  if write_err :
    try : 
      makedirs(f"{output_path}/err/")
    except : pass
  try :
    makedirs(f"{output_path}/target/")
  except : pass




def extract_nc_file(directory, list_all=None, CLOUDS=None, UVAI=None, RADIANCE=None, IRRADIANCE=None , ISRF=None):
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
          else :
            if filename[13:15] == 'AE': 
                extract_uvai_file(directory, list_all, UVAI)

def extract_irradiance_file(directory, list_all, IRRADIANCE):
    filename = directory.split('/')[-1]
    data = Dataset(directory,'r')
    prefix = filename[13:19]
    print (filename )
    groups_str = ['OBSERVATIONS', 'INSTRUMENT']

    for gr in groups_str :
      #print (str(gr) + ', ' , end='')
      for bd in IRRADIANCE.keys() :
        tmp_A = data.groups[bd + '_IRRADIANCE'].groups['STANDARD_MODE'].groups[gr]
        for var in tmp_A.variables:
          if var in list_all : IRRADIANCE[bd][var] = tmp_A.variables[var]
def extract_radiance_file(directory, list_all, RADIANCE):
    filename = directory.split('/')[-1]
    data = Dataset(directory,'r')
    prefix = filename[13:15]
    #globals()['bd'] = 'BAND' + filename[18:19]
    bd = 'BAND' + filename[18:19]
    print (filename )
    groups_str = ['OBSERVATIONS', 'INSTRUMENT', 'GEODATA']

    #if geo_done == 0 : # retrieve geodata once
     # tmp_B = data.groups[ bd + '_RADIANCE'].groups['STANDARD_MODE'].groups['GEODATA']
     # print (str('GEODATA') + ', ' , end = '')
     # for var in tmp_B.variables:
     #   globals()[var] = tmp_B.variables[var][:]
     # globals()['geo_done'] = 1
      
    for gr in groups_str :
      tmp_A = data.groups[ bd + '_RADIANCE'].groups['STANDARD_MODE'].groups[gr]
      #print (str(gr) + ', ', end='' )
      for var in tmp_A.variables:
        if var in list_all : RADIANCE[ bd][var] = tmp_A.variables[var]

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

def extract_uvai_file(directory, list_all, UVAI):
    filename = directory.split('/')[-1]
    data = Dataset(directory, 'r')
    prefix = filename[13:15]
    print (filename )
    vars = ['aerosol_index_340_380', 'aerosol_index_354_388' ] 
    for var in vars :
      UVAI[var] = data.groups['PRODUCT'].variables[var]

def extract_isrf_file(directory, ISRF):
    filename = directory.split('/')[-1] 
    data = Dataset( directory, 'r')
    conv_bd = {'BAND1':'band_1','BAND2':'band_2',
             'BAND3':'band_3','BAND4':'band_4',
             'BAND5':'band_5','BAND6':'band_6'}
    print(filename)
    for bd in ISRF.keys() :
      for var in data[conv_bd[bd]].variables:
        ISRF[bd][var] = data[conv_bd[bd]][var]

def write_pixel(i, j, STD, RADIANCE, IRRADIANCE, CLOUDS, ISRF, do_isrf, 
                do_l1b, do_err, do_target, output_path , 
                utc_time, irregular, increment=None, output_path2=None, pixnum=None) :


  date = utc_time.strftime('%Y%m%d')
  NA = 6.022140857e20 # Avogadro/1000 for conversion from mol/m/nm/sr/s to photon/s/cm2/nm/sr

  #if pixnum % 500 == 0 : print(pixnum, end=' ', flush=True)
  wfilename = f'S5P_L1B_4ch_{date}_{str(pixnum)}.dat'
  w2filename = f'TargetSceneAttributes_kopr_{date}_{str(pixnum)}.asc'
  w3filename = f'S5P_ERR_4ch_{date}_{str(pixnum)}.dat'
 
  if do_l1b : write_l1b(i, j, output_path, wfilename, STD, RADIANCE, IRRADIANCE, NA, irregular)
 
  if do_err : write_err(RADIANCE, IRRADIANCE, i, j, output_path, w3filename, NA, irregular)
 
  if do_target : write_target(STD, CLOUDS, i, j, output_path, w2filename, utc_time)

  if do_isrf : write_isrf(ISRF, RADIANCE, i, j, output_path, pixnum, increment, pix, domain_shape, irregular)
  
  if do_l1b and do_target and do_err :
    with open(f'{output_path}/{pixnum}.UTtime' , 'w') as f1 :
      f1.write(utc_time.strftime('    %Y    %m    %d    %H    %M    %S' ))
    with open(f'{output_path}/{pixnum}.latlong', 'w') as f2 :
      f2.write(f"{STD['latitude_domain'][i]}    {STD['longitude_domain'][i]}")

  return pixnum


def write_isrf(ISRF, RADIANCE, i, j, output_path, pixnum, increment, pix, domain_shape, irregular): 
  """Not working"""
  isrf_out_path = f'{output_path}/isrf/s5p_isrf_{pixnum}.bin' if increment is None else f'{output_path}/../isrf/s5p_isrf_{pixnum}.bin'
  if pix <= (domain_shape[1]-1) and not os.path.exists(isrf_out_path) :
    with open(isrf_out_path, 'wb' ) as isrf_file : # ISRF file
      for bd in ISRF.keys():
        if bd in irregular : np.float64(RADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i]),:]).data.tofile(isrf_file)
        else : np.float64(RADIANCE[bd]['nominal_wavelength_domain'][:].data[j,:]).tofile(isrf_file)
        if  bd in irregular : np.float64(ISRF[bd]['isrf_domain_resampled0'][int(RADIANCE[bd]['lon_ref'][0,i]),:,:]).tofile(isrf_file)
        else : np.float64(ISRF[bd]['isrf_domain_resampled0'][:][j,:,:]).tofile(isrf_file)
        np.float64(ISRF[bd]['delta_wavelength'][:].data).tofile(isrf_file)

def write_target(STD, CLOUDS, i, j, output_path, w2filename, utc_time) :
  #if j < 225 :
  #   vaa = - STD['viewing_azimuth_angle_domain'][i] + 180 
  #else : 
  #   vaa = STD['viewing_azimuth_angle_domain'][i] 
  #relaz = STD['SAA_vlidort'][i] - vaa
  relaz = STD['relaz'][i]

  header5 = 'Cloud_top_pressure\tEffetive_cloud_fraction\tErr_Cloud_top_pressure\tErr_effective_cloud_fraction'\
            '\tCloud_albedo\tSurface_albedo\tSurface_pressur\n'
  header3 = 'TES_File_ID = "L2: Target_Scene_Attributes\n'\
          'Data_size = 1 x 15\n'\
          'Dayflag = 0\n'\
          f'UtcTime = {utc_time.strftime("%Y-%m-%dT")}'\
          f'{utc_time.strftime("%H:%M:%S")}.000000Z\n'\
          'End_of_Header  ****  End_of_Header  ****  End_of_Header\n'\
          'View_Mode Latitude\tLongitude\tFoot_Print_Mean_Height\tOrbit_Inclination_Angle\tSC_Altitude\t'\
          'SC_Latitude\tBoresight_Angle\tBoresight_Angle_Uncertainty\tSpacecraft_Azimuth\tTarget_Radius\t'\
          'Satellite_Radius\tSZA\tVZA\tAZ\n'\
          'N/A\tDegrees\tDegrees\tMeters\tDegrees\tMeters\tDegrees\tRadians\tRadians\tRadians\tMeters\tMeters\t(deg)\t(deg)\t(deg)\n'

          #f'{date[20:24]}-{date[24:26]}-{date[26:28]}T'\
          #f'{date[29:31]}:{date[31:33]}:{date[33:35]}.000000Z\n'\



  with open(f"{output_path}/target/{w2filename}", 'a') as tget : # TargetScene file
    tget.write(header3)
    tget.write(f"Nadir\t{STD['latitude_domain'][i]}\t{STD['longitude_domain'][i]}\t0.0\t0.0\t{STD['satellite_altitude_domain'][i]}\t"\
            f"{STD['satellite_latitude_domain'][i]}\t0.0\t0.0\t{STD['VAA_vlidort'][i]}\t0.0\t0.0\t"\
            f"{STD['solar_zenith_angle_domain'][i]}\t{STD['viewing_zenith_angle_domain'][i]}\t{relaz}\n")
    tget.write('\n'+header5)
    tget.write(f"{CLOUDS['cloud_top_pressure_domain_resampled'][i]}\t{CLOUDS['cloud_fraction_domain_resampled'][i]}\t{CLOUDS['cloud_top_pressure_precision_domain_resampled'][i]}\t"\
            f"{CLOUDS['cloud_fraction_precision_domain_resampled'][i]}\t{CLOUDS['cloud_albedo_crb_domain_resampled'][i]}\t{CLOUDS['surface_albedo_fitted_domain_resampled'][i]}\t" \
            f"{CLOUDS['surface_pressure_domain_resampled'][i]}")

def write_err(RADIANCE, IRRADIANCE, i, j, output_path2, w3filename, NA, irregular) :
  with open(f"{output_path2}/err/{w3filename}", 'w') as err : # S5P_ERR file
      for bd in RADIANCE.keys() :
        if bd in irregular : (NA*RADIANCE[bd]['radiance_error_domain_resampled'][i,:]).tofile(err, sep='\t', format='%1.4e');err.write('\n') #resampled
        else : (NA*RADIANCE[bd]['radiance_error_domain'][i,:]).tofile(err, sep='\t', format='%1.4e');err.write('\n')
        if bd in irregular : (NA*IRRADIANCE[bd]['irradiance_error_domain'][RADIANCE[bd]['ground_pixel_domain'][int(RADIANCE[bd]['idx'][i])],:]).tofile(err, sep='\t', format='%1.4e');err.write('\n')
        else : (NA*IRRADIANCE[bd]['irradiance_error_domain'][j,:]).tofile(err, sep='\t', format='%1.4e');err.write('\n')

def write_l1b(i, j, output_path2, wfilename, STD, RADIANCE, IRRADIANCE, NA, irregular):
  header1 = 'Latitude \t Longitude \t n_lambda_rad_ch1 \t n_lambda_rad_ch2 \t n_lambda_rad_ch3 \t n_lambda_rad_ch4 \t' \
          'n_lambda_ir_ch1 \t n_lambda_irr_ch2 \t n_lambda_irr_ch3 \t n_lambda_irr_ch4 \t swath_number\n'

  header2 = 'lambda_rad_ch1(nm) ; rad_ch1 (photon/s/cm2/nm/sr) ; lambda_irrad_ch1 ; '\
          'irrad_ch1 ; lambda_rad_ch2 ; rad_ch2 ; lambda_irrad_ch2 ; irrad_ch2 .....\n'
  with open(f"{output_path2}/l1b/{wfilename}", 'w') as f1 : # S5P_L1B file
    f1.write(header1)
    f1.write(f"{STD['latitude_domain'][i]}\t{STD['longitude_domain'][i]}\t")
    for bd in RADIANCE.keys() :
      f1.write(f"{RADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
    for bd in IRRADIANCE.keys() :
      f1.write(f"{IRRADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
    f1.write(f"{j}\n{header2}") #WARNING swath starts from 0
    for bd in RADIANCE.keys() :
      if bd in irregular : RADIANCE[bd]['nominal_wavelength_domain'][RADIANCE[bd]['ground_pixel_domain'][int(RADIANCE[bd]['idx'][i])],:].tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : RADIANCE[bd]['nominal_wavelength_domain'][:][j,:].tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : (NA*RADIANCE[bd]['radiance_domain_resampled'][i,:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n') #resampled
      else : (NA*RADIANCE[bd]['radiance_domain'][i,:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : IRRADIANCE[bd]['nominal_wavelength_domain'][RADIANCE[bd]['ground_pixel_domain'][int(RADIANCE[bd]['idx'][i])],:].tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : IRRADIANCE[bd]['nominal_wavelength_domain'][j,:].tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : (NA*IRRADIANCE[bd]['irradiance_domain'][RADIANCE[bd]['ground_pixel_domain'][int(RADIANCE[bd]['idx'][i])],:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : (NA*IRRADIANCE[bd]['irradiance_domain'][j,:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n')

def append_param(list_param, STD, CLOUDS, i, j, pixnum, utc_time) : 
  #if j < 225 :
  #   vaa = - STD['viewing_azimuth_angle_domain'][i] + 180 
  #else : 
  #   vaa = STD['viewing_azimuth_angle_domain'][i] 
  #relaz = STD['SAA_vlidort'][i] - vaa
  relaz = STD['relaz'][i]

  header4 = '%pixnum\t lon_centre\t lat_centre\t cloud_fraction\t '\
              'lat_bounds0\t lat_bounds1\t lat_bounds2\t lat_bounds3\t lon_bounds0\t lon_bounds1\t lon_bounds2\t lon_bounds3\t '\
              'lon_sat\t lat_sat\t sat_alt\t rel_swat_pos\t elevation\t utc_time\t cloud_top_pressure\t '\
              'err_cloud_top_pressure\t err_cloud_fraction\t cloud_albedo\t surface_albedo\t surface_pressure\t '\
              'SZA2\tSZA2\tSZA2\t SAA2\t SAA2\tSAA2\tVZA2\t VZA2\tVZA2\tVAA2\t VAA2\tVAA2\t SZA_Vlidort\t VZA_Vlidort\t Relazimuth_vilidort\n'
  #if glob(f"{output_path}/list.s5p_param") == [] :
  #list_param.write(header4)
  list_param.write(f"{pixnum}\t{STD['longitude_domain'][i]}\t{STD['latitude_domain'][i]}\t{CLOUDS['cloud_fraction_domain_resampled'][i]}\t"\
          f"{STD['longitude_bounds_domain'][i,0]}\t{STD['longitude_bounds_domain'][i,1]}\t{STD['longitude_bounds_domain'][i,2]}\t{STD['longitude_bounds_domain'][i,3]}\t"\
          f"{STD['latitude_bounds_domain'][i,0]}\t{STD['latitude_bounds_domain'][i,1]}\t{STD['latitude_bounds_domain'][i,2]}\t{STD['latitude_bounds_domain'][i,3]}\t"\
          f"{STD['satellite_longitude_domain'][i]}\t{STD['satellite_latitude_domain'][i]}\t{STD['satellite_altitude_domain'][i]}\t"\
          f"{str(j)}\t{CLOUDS['surface_altitude_domain_resampled'][i]}\t"\
          f"{utc_time.timestamp()}\t{CLOUDS['cloud_top_pressure_domain_resampled'][i]}\t"\
          f"{CLOUDS['cloud_top_pressure_precision_domain_resampled'][i]}\t{CLOUDS['cloud_fraction_precision_domain_resampled'][i]}\t"\
          f"{CLOUDS['cloud_albedo_crb_domain_resampled'][i]}\t{CLOUDS['surface_albedo_fitted_domain_resampled'][i]}\t"\
          f"{CLOUDS['surface_pressure_domain_resampled'][i]}\t"\
          f"{STD['solar_zenith_angle_domain'][i]}\t{STD['solar_zenith_angle_domain'][i]}\t{STD['solar_zenith_angle_domain'][i]}\t"\
          f"{STD['SAA_vlidort'][i]}\t{STD['SAA_vlidort'][i]}\t{STD['SAA_vlidort'][i]}\t"\
          f"{STD['viewing_zenith_angle_domain'][i]}\t{STD['viewing_zenith_angle_domain'][i]}\t{STD['viewing_zenith_angle_domain'][i]}\t"\
          f"{STD['VAA_vlidort'][i]}\t{STD['VAA_vlidort'][i]}\t{STD['VAA_vlidort'][i]}\t"\
          f"{STD['solar_zenith_angle_domain'][i]}\t{STD['viewing_zenith_angle_domain'][i]}\t{relaz}\n") # list.s5p_param  WARNING swath starts from 0
