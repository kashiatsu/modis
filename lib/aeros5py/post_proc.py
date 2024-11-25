import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
def read_output_files2(output_path) :
  """lon lat in aod dict"""
  df=pd.read_table(output_path+'/log_a_AOD_vlidort_all' , header=None, delim_whitespace=True, na_values='-9.99000E+0002', skipinitialspace=True,) #low_memory=False) #ok na_values=-9999,
  df = df.sort_values(df.columns[0], ascending=True, ignore_index=True)
  def make_header(head, nlev):
    a=np.arange(nlev)
    hlev=[]
    for i in a :
      hlev.append(head[-1]+str(a[i]+1))
    return head + hlev
  kC = ['pix', 'RMS', 'DOF', 'c0'] #log_a_O3_all #log_a_errtot_all #log_a_errmes_all #avk_a_O3_all
  kPT = ['pix', 'nlev', 'z0', 'p0'] #, 't0'] # log_a_pt_all
  kTS = ['pix', 'RMS', 'ts', 'tp'] # log_a_ts_all
  kN = ['pix', 'nlev1', 'z0', 'r0', 's0', 'n0'] #log_a_N_vlidort_all
  #kAOD = ['pix', 'nlev1', 'nlambda1', 'nspecies', 'z0:'] #log_a_AOD_vlidort_all
  a=make_header(kPT, 50)
  #N=pd.read_table(output_path+'/log_a_N_vlidort_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(a+['t0'], 50) ) #OK
  #avk=pd.read_table(output_path+'/avk_a_O3_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #ts=pd.read_table(output_path+'/log_a_ts_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kTS, 0) ) #ok
  o3=pd.read_table(output_path+'/log_a_O3_all', delim_whitespace=True, skipinitialspace=True, na_values='-9.99000E+0002', names=make_header(kC, 51) ) #ok
  o3 = o3.sort_values(o3.columns[0], ascending=True, ignore_index=True)
  #errt=pd.read_table(output_path+'/log_a_errtot_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #errm=pd.read_table(output_path+'/log_a_errmes_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #pt=pd.read_table(output_path+'/log_a_pt_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kPT, 99) ) #ok

  #cherry picking pixels => TODO add long lat to log_a_* files
  #if len(glob(output_path + '/absent_pixels*')) == 2 :
  #  absent_pixels = []
  #  with open(output_path + '/absent_pixels_o3') as f :
  #    absent_pixels = list(np.asarray(f.readline().split(), dtype=int))
  #  o3.drop(absent_pixels, inplace=True)
  #  o3.reset_index(inplace=True, drop=True)

  #  absent_pixels = []
  #  with open(output_path + '/absent_pixels_aod') as f :
  #    absent_pixels = list(np.asarray(f.readline().split(), dtype=int))
  #  df.drop(absent_pixels, inplace=True)
  #  df.reset_index(inplace=True, drop=True)
  #else :
  #  absent_pixels = []
  #  for i in range(o3.iloc[:,0].size) :
  #    if o3.iloc[i,0] not in list(df.iloc[:,0]) :
  #      absent_pixels.append(i)
  #  a=np.asarray(absent_pixels)
  #  a.tofile(output_path+'/absent_pixels_o3', sep=' ')
  #  o3.drop(absent_pixels, inplace=True)
  #  o3.reset_index(inplace=True, drop=True)

  #   
  #  absent_pixels = []
  #  for i in range(df.iloc[:,0].size) :
  #    if df.iloc[i,0] not in list(o3.iloc[:,0]) :
  #      absent_pixels.append(i)
  #  a=np.asarray(absent_pixels)
  #  a.tofile(output_path+'/absent_pixels_aod', sep=' ')
  #  df.drop(absent_pixels, inplace=True)
  #  df.reset_index(inplace=True, drop=True)
  #  print(df.shape)
  #  print(o3.shape)

  #pd.set_option('display.max_rows', None)
  #print (df.iloc[3968,:])
  nlev = int(df.iloc[0,1])
  nlambdas = int(df.iloc[0,2] )
  nspecies = int(df.iloc[0,3])
  aod = dict()

  aod['pix'] = df.iloc[:,0]
  aod['lon'] = df.iloc[:,-2]
  aod['lat'] = df.iloc[:,-1]
  aod['z0'] = df.iloc[:,4]
  aod['lambdas'] = np.zeros(nlambdas) * np.nan
  aod['aod'] = np.zeros([df.shape[0],nlambdas]) * np.nan
  aod['saod'] = np.zeros([df.shape[0],nlambdas]) * np.nan
  aod['N_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan
  aod['aero_ext'] = np.zeros([df.shape[0],nlev,nspecies]) * np.nan
  aod['aero_sca'] = np.zeros([df.shape[0],nlev,nspecies]) * np.nan
  aod['aod_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan
  aod['saod_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan
  #new
  aod['RMS'] = o3['RMS']
  aod['DOF'] = o3['DOF']

  for i in range(df.shape[0]): # for each pixel
    tmp = np.array(df.iloc[i,5:5+nlambdas*3]).reshape(nlambdas,3).transpose()
    aod['lambdas'] = tmp[0,:]
    aod['aod'][i,:] = tmp[1,:]
    aod['saod'][i,:] = tmp[2,:]



  j=4+(nlambdas)*3;
  for k in range(nspecies) :
      try :
        aod['N_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :
        print("WARNING: N_layer set to nan"+str(j))
        #aod['N_layer'][:,:,k] = np.nan
      j=j+nlev-1;
      try :
        aod['aero_ext'][:,:,k] = df.iloc[:,(j+nlev): j: -1];
      except :
        print("WARNING: aero_ext set to nan"+str(j))
        #aod['aero_ext'][:,:,k] = np.nan
      j=j+nlev;
      try :
        aod['aero_sca'][:,:,k] = df.iloc[:,(j+nlev):j: -1];
      except :
        print("WARNING: aero_sca set to nan"+str(j))
        #aod['aero_sca'][:,:,k] = np.nan;
      j=j+nlev;
      try :
         aod['aod_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :
        print("WARNING: aod_layer set to nan"+str(j))
        #aod['aod_layer'][:,:,k] = np.nan;
      j=j+nlev-1;
      try :
        aod['saod_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :
        print("WARNING: aod set to nan"+str(j))
        #aod['saod_layer'][:,:,k] = np.nan;

  #del df
  #gc.collect()
  return aod


def read_output_files(output_path) :
  df=pd.read_table(output_path+'/log_a_AOD_vlidort_all' , header=None, delim_whitespace=True, skipinitialspace=True,) #low_memory=False) #ok na_values=-9999,
  df = df.sort_values(df.columns[0], ascending=True, ignore_index=True)
  def make_header(head, nlev):
    a=np.arange(nlev)
    hlev=[]
    for i in a :
      hlev.append(head[-1]+str(a[i]+1))
    return head + hlev
  kC = ['pix', 'RMS', 'DOF', 'c0'] #log_a_O3_all #log_a_errtot_all #log_a_errmes_all #avk_a_O3_all
  kPT = ['pix', 'nlev', 'z0', 'p0'] #, 't0'] # log_a_pt_all
  kTS = ['pix', 'RMS', 'ts', 'tp'] # log_a_ts_all
  kN = ['pix', 'nlev1', 'z0', 'r0', 's0', 'n0'] #log_a_N_vlidort_all
  #kAOD = ['pix', 'nlev1', 'nlambda1', 'nspecies', 'z0:'] #log_a_AOD_vlidort_all
  a=make_header(kPT, 50)
  #N=pd.read_table(output_path+'/log_a_N_vlidort_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(a+['t0'], 50) ) #OK
  #avk=pd.read_table(output_path+'/avk_a_O3_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #ts=pd.read_table(output_path+'/log_a_ts_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kTS, 0) ) #ok
  o3=pd.read_table(output_path+'/log_a_O3_all', delim_whitespace=True, skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  o3 = o3.sort_values(o3.columns[0], ascending=True, ignore_index=True)
  #errt=pd.read_table(output_path+'/log_a_errtot_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #errm=pd.read_table(output_path+'/log_a_errmes_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kC, 51) ) #ok
  #pt=pd.read_table(output_path+'/log_a_pt_all', sep=' ', skipinitialspace=True, na_values=-9999, names=make_header(kPT, 99) ) #ok

  #cherry picking pixels => TODO add long lat to log_a_* files
  if len(glob(output_path + '/absent_pixels*')) == 2 :
    absent_pixels = []
    with open(output_path + '/absent_pixels_o3') as f :
      absent_pixels = list(np.asarray(f.readline().split(), dtype=int))
    o3.drop(absent_pixels, inplace=True)
    o3.reset_index(inplace=True, drop=True)

    absent_pixels = []
    with open(output_path + '/absent_pixels_aod') as f :
      absent_pixels = list(np.asarray(f.readline().split(), dtype=int))
    df.drop(absent_pixels, inplace=True)
    df.reset_index(inplace=True, drop=True)
  else :
    absent_pixels = []
    for i in range(o3.iloc[:,0].size) :
      if o3.iloc[i,0] not in list(df.iloc[:,0]) :
        absent_pixels.append(i)
    a=np.asarray(absent_pixels)
    a.tofile(output_path+'/absent_pixels_o3', sep=' ')
    o3.drop(absent_pixels, inplace=True)
    o3.reset_index(inplace=True, drop=True)

     
    absent_pixels = []
    for i in range(df.iloc[:,0].size) :
      if df.iloc[i,0] not in list(o3.iloc[:,0]) :
        absent_pixels.append(i)
    a=np.asarray(absent_pixels)
    a.tofile(output_path+'/absent_pixels_aod', sep=' ')
    df.drop(absent_pixels, inplace=True)
    df.reset_index(inplace=True, drop=True)
    print(df.shape)
    print(o3.shape)

  #pd.set_option('display.max_rows', None)
  #print (df.iloc[3968,:])
  nlev = int(df.iloc[0,1])
  nlambdas = int(df.iloc[0,2] ) 
  nspecies = int(df.iloc[0,3])
  aod = dict()

  aod['pix'] = df.iloc[:,0]
  aod['z0'] = df.iloc[:,4]
  aod['lambdas'] = np.zeros(nlambdas) * np.nan
  aod['aod'] = np.zeros([df.shape[0],nlambdas]) * np.nan
  aod['saod'] = np.zeros([df.shape[0],nlambdas]) * np.nan
  aod['N_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan 
  aod['aero_ext'] = np.zeros([df.shape[0],nlev,nspecies]) * np.nan
  aod['aero_sca'] = np.zeros([df.shape[0],nlev,nspecies]) * np.nan
  aod['aod_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan
  aod['saod_layer'] = np.zeros([df.shape[0],nlev-1,nspecies]) * np.nan
  #new
  aod['RMS'] = o3['RMS']
  aod['DOF'] = o3['DOF']
  


  for i in range(df.shape[0]): # for each pixel
    tmp = np.array(df.iloc[i,5:5+nlambdas*3]).reshape(nlambdas,3).transpose()
    aod['lambdas'] = tmp[0,:]
    aod['aod'][i,:] = tmp[1,:]
    aod['saod'][i,:] = tmp[2,:]



  j=4+(nlambdas)*3;
  for k in range(nspecies) : 
      try :
        aod['N_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :
        print("WARNING: N_layer set to nan"+str(j))
        #aod['N_layer'][:,:,k] = np.nan
      j=j+nlev-1;
      try :
        aod['aero_ext'][:,:,k] = df.iloc[:,(j+nlev): j: -1];
      except :
        print("WARNING: aero_ext set to nan"+str(j))
        #aod['aero_ext'][:,:,k] = np.nan
      j=j+nlev;
      try :
        aod['aero_sca'][:,:,k] = df.iloc[:,(j+nlev):j: -1];
      except :
        print("WARNING: aero_sca set to nan"+str(j))
        #aod['aero_sca'][:,:,k] = np.nan;
      j=j+nlev;
      try :
         aod['aod_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :
        print("WARNING: aod_layer set to nan"+str(j))
        #aod['aod_layer'][:,:,k] = np.nan;
      j=j+nlev-1;
      try :
        aod['saod_layer'][:,:,k] = df.iloc[:,(j+nlev-1): j: -1];
      except :  
        print("WARNING: aod set to nan"+str(j))
        #aod['saod_layer'][:,:,k] = np.nan;

  #del df
  #gc.collect()
  return aod

def read_param2(directory):
  h="pixnum1    lon_centre    lat_centre    cloud_fraction     lat_bounds0    lat_bounds1    lat_bounds2    lat_bounds3    lon_bounds0    lon_bounds1    lon_bounds2   lon_bounds3    lon_sat    lat_sat    sat_alt    swat_pos    elevation    utc_time    cloud_top_pressure    err_cloud_top_pressure    err_cloud_fraction    cloud_albedo    surface_albedo    surface_pressure    SZA1    SZA2    SZA3    SAA1    SAA2    SAA3    VZA1   VZA2   VZA3    VAA1   VAA2   VAA3    SZA_Vlidort    VZA_Vlidort    Relazimuth_vlidort"

  param = pd.read_csv(directory, skiprows=0, names=h.split(), delim_whitespace=True, index_col=False)
  return param

def read_param(directory, aod):
  h="pixnum1    lon_centre    lat_centre    cloud_fraction     lat_bounds0    lat_bounds1    lat_bounds2    lat_bounds3    lon_bounds0    lon_bounds1    lon_bounds2   lon_bounds3    lon_sat    lat_sat    sat_alt    swat_pos    elevation    utc_time    cloud_top_pressure    err_cloud_top_pressure    err_cloud_fraction    cloud_albedo    surface_albedo    surface_pressure    SZA1    SZA2    SZA3    SAA1    SAA2    SAA3    VZA1   VZA2   VZA3    VAA1   VAA2   VAA3    SZA_Vlidort    VZA_Vlidort    Relazimuth_vlidort"

  param = pd.read_csv(directory, skiprows=0, names=h.split(), delim_whitespace=True)
  a= list(aod['pix'])
  notFound = []
  for i, p in enumerate(param.iloc[:,0]) :
    if p not in a :
      notFound.append(i)
  param2 = param.drop(notFound)
  #coo = pd.DataFrame(param.iloc[:, 0:2], columns=['pix','lon','lat'])
  return param2


def read_coordinates(directory, aod):
  df2 = pd.read_csv(directory, delim_whitespace=True, skipinitialspace=True, header=None, na_values=-999)
  a = list(tuple())
  for i, pix in enumerate(df2.iloc[:,0]) :
    if pix in np.array(aod['pix']) :
      a.append((pix, df2.iloc[i,1] , df2.iloc[i,2])) 
  coo = pd.DataFrame(a, columns=['pix','lon','lat'])
  
  #print (coo.pix)

 # with open(f'{input_path}list.{sat}_param_iasipixels', 'r') as f1 :
 #   print(f'{input_path}list.{sat}_param_iasipixels')
 #   #f1.readline()
 #   for line in f1.readlines() :
 #     if line.split()[1] != 'NaN':
 #       if int(line.split()[0]) in np.array(o3.pix) :
 #         a.append((int(line.split()[0]), float(line.split()[1]), float(line.split()[2]))) 
 # coo = pd.DataFrame(a, columns=['pix','lon','lat'])
 # print (coo.pix)
  return coo






def read_CAL_L1(directory):
  from pyhdf.HDF import HDF
  from pyhdf.VS import VS
  data = xr.open_dataset(directory) 
  Lidar_Data_Altitude = np.asarray(HDF(directory).vstart().attach('metadata')[0][29], dtype=np.float)
  if Lidar_Data_Altitude.size < 300 :
    Lidar_Data_Altitude = np.asarray(HDF(directory).vstart().attach('metadata')[0][27], dtype=np.float)
  data['Lidar_Data_Altitude'] = Lidar_Data_Altitude
  return data

def convert_aod_wv(aod1, wv1, wv2, alpha) :
  """Converts AOD using angstrom exponent :
    Args : floats
      aod1(wv1), wv1, wv2, alpha

    return: float
      aod2(wv2)"""
  #print(f'\nConverted AOD from {wv1} nm to {wv2} nm') # using angstrom exp = {alpha.mean()}')
  return  aod1 * ( wv2/wv1 )**(-alpha)
