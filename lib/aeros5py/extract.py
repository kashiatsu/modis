#!/home/flemmouchi/.conda/envs/py36/bin/python3.6

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
def write_pixel(i,j, STD, RADIANCE, IRRADIANCE, CLOUDS, ISRF, do_isrf, do_l1b, do_err, do_target, output_path, header1,  header2,  header3, header4, header5 , date, irregular, increment=None, output_path2=None) :
  utc_time = 8
  NA = 6.022140857e20 # Avogadro/1000 for conversion from mol/m/nm/sr/s to photon/s/cm2/nm/sr
  domain_shape = STD['longitude_domain'].shape
  itmp = i

  if increment is not None : #batch True

    n = (i // increment)
    itmp = i - n*increment
    output_path=f"{output_path}/{str(n)}/"
    output_path2 = f"{output_path2}/{str(n)}/" if output_path2 is not None else output_path

  else : # batch False
    output_path2 = output_path

  pix = domain_shape[1] *itmp+j
  pixnum = 10001 + pix
  

  #if pixnum % 500 == 0 : print(pixnum, end=' ', flush=True)
  wfilename = f'S5P_L1B_4ch_{date}_{str(pixnum)}.dat'
  w2filename = f'TargetSceneAttributes_kopr_{date}_{str(pixnum)}.asc'
  w3filename = f'S5P_ERR_4ch_{date}_{str(pixnum)}.dat'

  with open(f"{output_path}/target/list.cloudfree", 'a') as list_cf :
    list_cf.write(f"{STD['longitude_domain'].data[i,j]}\t{STD['latitude_domain'].data[i,j]}\t0.0\t{wfilename}\n")

  append_param(output_path, STD, CLOUDS, i, j, pixnum, utc_time, header4) 
 
  if do_l1b : write_l1b(i, j, output_path2, wfilename, STD, RADIANCE, IRRADIANCE, NA, irregular, header2, header1)
 
  if do_err : write_err(RADIANCE, IRRADIANCE, i, j, output_path2, w3filename, NA, irregular)
 
  if do_target : write_target(STD, CLOUDS, i, j, output_path, header3, header5, w2filename)

  if do_isrf : write_isrf(ISRF, RADIANCE, i, j, output_path, pixnum, increment, pix, domain_shape, irregular)

def write_pixel_partially(i,j, STD, RADIANCE, IRRADIANCE, CLOUDS, ISRF, do_isrf, do_l1b, do_err, do_target, output_path, header1,  header2,  header3, header4, header5 , date, irregular, increment=None, output_path2=None, pixnum=None) :
  utc_time = 8
  NA = 6.022140857e20 # Avogadro/1000 for conversion from mol/m/nm/sr/s to photon/s/cm2/nm/sr
  domain_shape = STD['longitude_domain'].shape
  itmp = i

  if increment is not None : #batch True

    n = (i // increment)
    itmp = i - n*increment
    output_path=f"{output_path}/{str(n)}/"
    output_path2 = f"{output_path2}/{str(n)}/" if output_path2 is not None else output_path

  else : # batch False
    output_path2 = output_path

  pix = domain_shape[1] *itmp+j
  #pixnum = 10001 + pix  to be uncommented
  
  #write isrf anyways 
  if do_isrf : write_isrf(ISRF, RADIANCE, i, j, output_path, 10001 + pix , increment, pix, domain_shape, irregular)
  if pix % 15 == 0 : # (condition to reduce pixel number) to be removed
    pixnum = pixnum+1 
  else : return pixnum

  if pixnum % 500 == 0 : print(pixnum, end=' ', flush=True)
  wfilename = f'S5P_L1B_4ch_{date}_{str(pixnum)}.dat'
  w2filename = f'TargetSceneAttributes_kopr_{date}_{str(pixnum)}.asc'
  w3filename = f'S5P_ERR_4ch_{date}_{str(pixnum)}.dat'

  with open(f"{output_path}/target/list.cloudfree", 'a') as list_cf :
    list_cf.write(f"{STD['longitude_domain'].data[i,j]}\t{STD['latitude_domain'].data[i,j]}\t0.0\t{wfilename}\n")

  append_param(output_path, STD, CLOUDS, i, j, pixnum, utc_time, header4) 
 
  if do_l1b : write_l1b(i, j, output_path2, wfilename, STD, RADIANCE, IRRADIANCE, NA, irregular, header2, header1)
 
  if do_err : write_err(RADIANCE, IRRADIANCE, i, j, output_path2, w3filename, NA, irregular)
 
  if do_target : write_target(STD, CLOUDS, i, j, output_path, header3, header5, w2filename)
  return pixnum

def write_isrf(ISRF, RADIANCE, i, j, output_path, pixnum, increment, pix, domain_shape, irregular): 
  isrf_out_path = f'{output_path}/isrf/s5p_isrf_{pixnum}.bin' if increment is None else f'{output_path}/../isrf/s5p_isrf_{pixnum}.bin'
  if pix <= (domain_shape[1]-1) and not os.path.exists(isrf_out_path) :
    with open(isrf_out_path, 'wb' ) as isrf_file : # ISRF file
      for bd in ISRF.keys():
        # np.float64(ISRF[bd]['wavelength_domain_resampled'][j,:]).tofile(isrf_file)
        #np.float64(RADIANCE[bd]['nominal_wavelength_domain'][:].data[j,:]).tofile(isrf_file) #not used
        if bd in irregular : np.float64(RADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:]).data.tofile(isrf_file)
        else : np.float64(RADIANCE[bd]['nominal_wavelength_domain'][:].data[j,:]).tofile(isrf_file)
        if  bd in irregular : np.float64(ISRF[bd]['isrf_domain_resampled0'][int(RADIANCE[bd]['lon_ref'][0,i]),:,:]).tofile(isrf_file)
        else : np.float64(ISRF[bd]['isrf_domain_resampled0'][:][j,:,:]).tofile(isrf_file)
        np.float64(ISRF[bd]['delta_wavelength'][:].data).tofile(isrf_file)

def write_target(STD, CLOUDS, i, j, output_path, header3, header5, w2filename) :
  with open(f"{output_path}/target/{w2filename}", 'a') as tget : # TargetScene file
    tget.write(header3)
    tget.write(f"Nadir\t{STD['latitude_domain'][i,j]}\t{STD['longitude_domain'][i,j]}\t0.0\t0.0\t{STD['satellite_altitude_domain'][i,j]}\t"\
            f"{STD['satellite_latitude_domain'][i,j]}\t0.0\t0.0\t{STD['VAA_vlidort'][i,j]}\t0.0\t0.0\t"\
            f"{STD['solar_zenith_angle_domain'][i,j]}\t{STD['viewing_zenith_angle_domain'][i,j]}\t{STD['RELAZM_vlidort'][i,j]}\n")
    tget.write('\n'+header5)
    tget.write(f"{CLOUDS['cloud_top_pressure_domain_resampled'][i,j]}\t{CLOUDS['cloud_fraction_domain_resampled'][i,j]}\t{CLOUDS['cloud_top_pressure_precision_domain_resampled'][i,j]}\t"\
            f"{CLOUDS['cloud_fraction_precision_domain_resampled'][i,j]}\t{CLOUDS['cloud_albedo_crb_domain_resampled'][i,j]}\t{CLOUDS['surface_albedo_fitted_domain_resampled'][i,j]}\t" \
            f"{CLOUDS['surface_pressure_domain_resampled'][i,j]}")
def write_err(RADIANCE, IRRADIANCE, i, j, output_path2, w3filename, NA, irregular) :
  with open(f"{output_path2}/err/{w3filename}", 'w') as err : # S5P_ERR file
      #err.write(header1)
      #err.write(f"{STD['latitude_domain'].data[i,j]}\t{STD['longitude_domain'].data[i,j]}\t")
      #for bd in RADIANCE.keys() :
      #  err.write(f"{RADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
      #for bd in IRRADIANCE.keys() :
      #  err.write(f"{IRRADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
      #err.write(f"{lon_start+j}\n{header2}")
      #err.write(f"{j+1}\n{header2}")
      for bd in RADIANCE.keys() :
        #if bd in irregular : RADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:].data.tofile(err, sep='\t', format='%1.4e');err.write('\n')
        #else : RADIANCE[bd]['nominal_wavelength_domain'][:].data[j,:].tofile(err, sep='\t', format='%1.4e');err.write('\n')
        if bd in irregular : (NA*RADIANCE[bd]['radiance_error_domain_resampled'][i,j,:]).tofile(err, sep='\t', format='%1.4e');err.write('\n') #resampled
        else : (NA*RADIANCE[bd]['radiance_error_domain'].data[i,j,:]).tofile(err, sep='\t', format='%1.4e');err.write('\n')
        #if bd in irregular : IRRADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:].data.tofile(err, sep='\t', format='%1.4e');err.write('\n')
        #else : IRRADIANCE[bd]['nominal_wavelength_domain'][j,:].data.tofile(err, sep='\t', format='%1.4e');err.write('\n')
        if bd in irregular : (NA*IRRADIANCE[bd]['irradiance_error_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:]).data.tofile(err, sep='\t', format='%1.4e');err.write('\n')
        else : (NA*IRRADIANCE[bd]['irradiance_error_domain'][j,:].data).tofile(err, sep='\t', format='%1.4e');err.write('\n')
def write_l1b(i, j, output_path2, wfilename, STD, RADIANCE, IRRADIANCE, NA, irregular, header2, header1):
  with open(f"{output_path2}/l1b/{wfilename}", 'w') as f1 : # S5P_L1B file
    f1.write(header1)
    f1.write(f"{STD['latitude_domain'].data[i,j]}\t{STD['longitude_domain'].data[i,j]}\t")
    for bd in RADIANCE.keys() :
      f1.write(f"{RADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
    for bd in IRRADIANCE.keys() :
      f1.write(f"{IRRADIANCE[bd]['nominal_wavelength_domain'][:].shape[-1]}\t")
    #f1.write(str(lon_start+j) + '\n'+ header2)
    f1.write(f"{str(j+1)}\n{header2}")
    for bd in RADIANCE.keys() :
      if bd in irregular : RADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:].data.tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : RADIANCE[bd]['nominal_wavelength_domain'][:].data[j,:].tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : (NA*RADIANCE[bd]['radiance_domain_resampled'][i,j,:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n') #resampled
      else : (NA*RADIANCE[bd]['radiance_domain'].data[i,j,:]).tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : IRRADIANCE[bd]['nominal_wavelength_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:].data.tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : IRRADIANCE[bd]['nominal_wavelength_domain'][j,:].data.tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      if bd in irregular : (NA*IRRADIANCE[bd]['irradiance_domain'][int(RADIANCE[bd]['lon_ref'][i,j]),:]).data.tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
      else : (NA*IRRADIANCE[bd]['irradiance_domain'][j,:].data).tofile(f1, sep='\t', format='%1.4e');f1.write('\n')
def append_param(output_path, STD, CLOUDS, i, j, pixnum, utc_time, header4) : 
  with open(f"{output_path}/target/list.s5p_param", 'a') as list_param :
    #if glob(f"{output_path}/list.s5p_param") == [] :
    #list_param.write(header4)
    list_param.write(f"{pixnum}\t{STD['longitude_domain'][i,j]}\t{STD['latitude_domain'][i,j]}\t{CLOUDS['cloud_fraction_domain_resampled'][i,j]}\t"\
            f"{STD['longitude_bounds_domain'][i,j,0]}\t{STD['longitude_bounds_domain'][i,j,1]}\t{STD['longitude_bounds_domain'][i,j,2]}\t{STD['longitude_bounds_domain'][i,j,3]}\t"\
            f"{STD['latitude_bounds_domain'][i,j,0]}\t{STD['latitude_bounds_domain'][i,j,1]}\t{STD['latitude_bounds_domain'][i,j,2]}\t{STD['latitude_bounds_domain'][i,j,3]}\t"\
            f"{STD['satellite_longitude_domain'][i,j]}\t{STD['satellite_latitude_domain'][i,j]}\t{STD['satellite_altitude_domain'][i,j]}\t"\
            f"{str(j+1)}\t{CLOUDS['surface_altitude_domain_resampled'][i,j]}\t"\
            f"{int(utc_time)}\t{CLOUDS['cloud_top_pressure_domain_resampled'][i,j]}\t"\
            f"{CLOUDS['cloud_top_pressure_precision_domain_resampled'][i,j]}\t{CLOUDS['cloud_fraction_precision_domain_resampled'][i,j]}\t"\
            f"{CLOUDS['cloud_albedo_crb_domain_resampled'][i,j]}\t{CLOUDS['surface_albedo_fitted_domain_resampled'][i,j]}\t"\
            f"{CLOUDS['surface_pressure_domain_resampled'][i,j]}\t"\
            f"{STD['solar_zenith_angle_domain'][i,j]}\t{STD['solar_zenith_angle_domain'][i,j]}\t{STD['solar_zenith_angle_domain'][i,j]}\t"\
            f"{STD['SAA_vlidort'][i,j]}\t{STD['SAA_vlidort'][i,j]}\t{STD['SAA_vlidort'][i,j]}\t"\
            f"{STD['viewing_zenith_angle_domain'][i,j]}\t{STD['viewing_zenith_angle_domain'][i,j]}\t{STD['viewing_zenith_angle_domain'][i,j]}\t"\
            f"{STD['VAA_vlidort'][i,j]}\t{STD['VAA_vlidort'][i,j]}\t{STD['VAA_vlidort'][i,j]}\t"\
            f"{STD['solar_zenith_angle_domain'][i,j]}\t{STD['viewing_zenith_angle_domain'][i,j]}\t{STD['RELAZM_vlidort'][i,j]}\n") # list.s5p_param
