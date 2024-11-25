#!/bin/python

#hidden imports
import pkg_resources.py2_warn, cftime, six

import netCDF4 as nc
import numpy as np
from glob import glob
from aeros5py.utils import do_pooling
import datetime as dt

def main(date, lat1=-39.0,  lat2=-10.0 , lon1=110.0 , lon2=155.0, threshold=2, output_path=None, var='Integer_Cloud_Mask') :
  """
  Threshold : cloud mask threshold [0:3], 0:certainly cloudy, 3:certainly cloudfree
  var : either Integer_Cloud_Mask or Cloud_Optical_Thickness 
  (0 = cloudy, 1= probably cloudy, 2 = probably clear, 3 = confident clear, -1 = no result
    
  """
  print(locals())
  #path = '/DATA/SPECAT_2/farouk/VIIRS/AUSTRALIA/' 
  path = '/DATA/SPECAT_2/flemmouchi/VIIRS/GOLF_OF_GUINEA/' 
  #files = glob('/DATA/SPECAT_2/flemmouchi/VIIRS/20191222/*') 
  day = dt.datetime.strptime(date, "%Y%m%d").strftime("%j")
  print(day)
  files = glob(f"{path}/CLDMSK_L2_VIIRS_SNPP.A{date[0:4]}{day}*.nc") 
  if files == [] :
    print("no viirs files found")
    quit()
  lon1=float(lon1)
  lat1=float(lat1)
  lon2=float(lon2)
  lat2=float(lat2)
  threshold=float(threshold)
  
  #longitude = np.array(np.nan)
  #latitude = np.array(np.nan)
  #cloud_od = np.array(np.nan)

  longitude = np.empty(0)
  latitude = np.empty(0)
  cloud_od = np.empty(0)

  #flatten bin
  for file in files :
    print('Extracting: ', file)
    data = nc.Dataset(file, 'r')
    
    lon0 = data['geolocation_data']['longitude'][:]
    lat0 = data['geolocation_data']['latitude'][:]
    c0 = data['geophysical_data'][var][:]  


    lon, lat, c = do_pooling(lon0, lat0, c0, 'min', 32, 8) # octal
    print(lon.shape, lat.shape, c.shape)

    if var == 'Integer_Cloud_Mask' :
      domain = ( lon > lon1 ) * (lon < lon2 ) * ( lat > lat1 ) * ( lat < lat2 ) * ( c >=threshold ) #for cloudmask file
    elif var == 'Cloud_Optical_Thickness' and threshold == None :
      domain = ( lon > lon1 ) * (lon < lon2 ) * ( lat > lat1 ) * ( lat < lat2 ) #* c.mask #for clouds file
    elif var == 'Cloud_Optical_Thickness' and threshold != None :
      domain = ( lon > lon1 ) * (lon < lon2 ) * ( lat > lat1 ) * ( lat < lat2 ) #* (c <= threshold) #for clouds file
    #print(domain); import sys; sys.exit()

    lon = lon[domain]
    lat = lat[domain]
    c = c[domain]
 

    longitude = np.append(longitude, lon.flatten())
    latitude = np.append(latitude, lat.flatten())
    cloud_od = np.append(cloud_od, c.flatten())

  if output_path == None : 
    return longitude , latitude, cloud_od
  elif output_path == "default" :
    output_path = path  
    f = open(f'{output_path}/list.viirs_cloudfree_{date}.bin', 'wb')
  else :
    f = open(f'{output_path}/list.viirs_cloudfree_{date}.bin', 'wb')
  
  np.int32(longitude.size-1).tofile(f)
  np.float32(longitude[1:].data).tofile(f)
  np.float32(latitude[1:].data).tofile(f)
  np.float32(cloud_od[1:].data).tofile(f)

if __name__ == '__main__' :
  import sys
  if len(sys.argv) == 1 :

    print("""Usage: 
             extract_viirs date latmin latmax lonmin lonmax cloud_threshold output_path cloud_product\n
             Example :\n
             extract_viirs 20191222 -39.0 -10.0 110.0 155.0 2 here Integer_Cloud_Mask""")
  main(*sys.argv[1:])


def read_viirs_cloudfree(directory) :
  #with open(directory , 'rb') as f :
  #  bd = f.read()
  #n = np.frombuffer(bd, dtype=np.integer, count=1); n = int(n)
  #lon = np.frombuffer(bd, dtype=np.float32, offset=4,  count=n)
  #lat = np.frombuffer(bd, dtype=np.float32, offset=4+4*n,  count=n)
  #cm = np.frombuffer(bd, dtype=np.float32, offset=4+8*n,  count=n)

  with open(directory, 'rb') as f :
    n = np.int.from_bytes(f.read(4), byteorder="little")
    print(n, " elements")
    longitude = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=0) 
    latitude = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=4) 
    cloud_od = np.frombuffer(f.read(4*n), count=n, dtype=np.float32) #, offset=4) 
  return longitude, latitude, cloud_od
