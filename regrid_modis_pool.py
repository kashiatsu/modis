## %matplotlib inline 
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#from aeros5py import cartopyaxes
import numpy as np
import xarray as xr
#from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator,
from scipy.interpolate import griddata
import datetime as dt
from glob import glob
from aeros5py.utils import count_days
from functions import *
from multiprocessing import Pool 
import sys
from sklearn.neighbors import NearestNeighbors

from netCDF4 import Dataset
import h5py

#which_modis = "TERRA"
which_modis = "AQUA"
#which_modis = "BOTH"

variables = {
             #"dblue_aod55_land_best":"Deep_Blue_Aerosol_Optical_Depth_550_Land_Best_Estimate",
             "AOD_550_Dark_Target_Deep_Blue_Combined":"AOD_550_Dark_Target_Deep_Blue_Combined",
             #"Aerosol_Type_Land":"Aerosol_Type_Land",
            }
#res=0.2 #spatial resolution in degrees
hmin=0 #min altitude index chimere
hmax=15 #max altitude index chiemre
#projection=ccrs.PlateCarree()
#interp_routine = NearestNDInterpolator
#interp_routine = LinearNDInterpolator

#dates = ['20191224']
#dates  = count_days('20191201', '20200131', '%Y%m%d')
#dates  = count_days('20191201', '20191201', '%Y%m%d')

#for date in dates :
def do_date(date, res, which_modis, region) :
  print(date)
  pdate = dt.datetime.strptime(date, "%Y%m%d")

  if which_modis == 'BOTH' :
    modis_root = f"./data/{region}/MODIS/"
    modis_files = glob(f'{modis_root}/MYD04_L2_nc/M*{pdate.strftime("%Y")}{pdate.strftime("%j")}.??????????????????????.hdf')
  else : 
    modis_root = f"./data/{region}/MODIS/"

    print(f'{pdate.strftime("%Y")}{pdate.strftime("%j")}')
    modis_files = glob(f'{modis_root}/MYD04_L2_nc/M*{pdate.strftime("%Y")}{pdate.strftime("%j")}.*')
    
  print(modis_root)
  print(modis_files)
      
  combined_granules = {"longitude":np.empty(0),
                       "latitude":np.empty(0),
                      }
  for var in variables :
    combined_granules[var] = np.empty(0) 

  for modis_file in modis_files :
      modis_ds = xr.open_dataset(modis_file, engine="netcdf4")
      print(modis_file)
      for var in variables :
        if var == "AOD_550_Dark_Target_Deep_Blue_Combined" : 
          #tmp = modis_ds[variables[var]].where( (modis_ds[variables[var]] > 0) & (modis_ds["AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag"] == 3) & (~modis_ds[variables[var]].isnull().data) )
          #modis_ds['AOD_550_Dark_Target_Deep_Blue_Combined'] = modis_ds["AOD_550_Dark_Target_Deep_Blue_Combined"].clip(0,100)
          #tmp = modis_ds[variables[var]].where(modis_ds["AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag"] == 3)
          tmp = modis_ds[variables[var]].where( (modis_ds[variables[var]] > 0) & (modis_ds["AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag"] >=2) )
          combined_granules[var] = np.append( combined_granules[var], tmp.data[~tmp.isnull()] )
        else :
          combined_granules[var] = np.append( combined_granules[var], modis_ds[variables[var]].data[~tmp.isnull().data])
      combined_granules['longitude'] = np.append( combined_granules['longitude'], modis_ds.Longitude.data[~tmp.isnull().data])
      combined_granules['latitude'] = np.append( combined_granules['latitude'], modis_ds.Latitude.data[~tmp.isnull().data])
#      combined_granules['longitude'] = np.append( combined_granules['longitude'], modis_ds.Longitude.data[ ~modis_ds[variables[var]].isnull()])
#      combined_granules['latitude'] = np.append( combined_granules['latitude'], modis_ds.Latitude.data[ ~modis_ds[variables[var]].isnull()])
#      combined_granules['longitude'] = np.append( combined_granules['longitude'], modis_ds.Longitude.data.flatten())
#      combined_granules['latitude'] = np.append( combined_granules['latitude'], modis_ds.Latitude.data.flatten())

      modis_ds.close()
  out_modis_ds = xr.Dataset()
  combined_granules = xr.Dataset(combined_granules)

  print(modis_ds)

  # Mean pooling
  lonlat = np.asarray([combined_granules.longitude.data, combined_granules.latitude.data]).T
  radius = 0.5 * np.sqrt(2*(res**2))
  lon_neig = NearestNeighbors(n_neighbors=50, radius=radius).fit(lonlat)
  dist, neig = lon_neig.kneighbors()

  out_modis_dsd = dict()
  for modis_var in variables :
    #out_modis_ds[modis_var] = (['south_north', 'west_east'],np.empty(XX.shape)*np.nan)
    out_modis_dsd[modis_var] = np.empty(XX.shape)*np.nan

  #print("XX : ", XX[0,:])
  #print("YY : ", YY[:,0])
  for i in range(XX.shape[1]-1) :
      for j in range(YY.shape[0]-1) :
          center = ((XX[j,i] + XX[j,i+1])/2,(YY[j,i]+ YY[j+1,i])/2)
          a  =(combined_granules.longitude.data - center[0])**2 + (combined_granules.latitude.data - center[1])**2  
          argmin = (np.argmin(a))
          b = combined_granules.isel(longitude=neig[argmin,:], latitude=neig[argmin,:], AOD_550_Dark_Target_Deep_Blue_Combined=neig[argmin,:])
          mask = ((b.longitude.data >= XX[j,i]) & (b.longitude.data <= XX[j,i+1]) & (b.latitude.data >= YY[j, i]) & (b.latitude.data <= YY[j+1,i]))
          
          for modis_var in variables :
              #out_modis_dsd[modis_var][j,i] = np.nanmean(b[modis_var].data) #[mask])
              out_modis_dsd[modis_var][j,i] = np.nanmean(b[modis_var].data[mask])
              #out_lon_ds[j,i] = np.nanmean(b.longitude.data[mask])
              #out_lat_ds[j,i] = np.nanmean(b.latitude.data[mask])

          #at the borders
          if (i == XX.shape[1]-2):
            center = ((XX[j,i+1] + res/2),(YY[j,i+1]+ res/2))
            a  =(combined_granules.longitude.data - center[0])**2 + (combined_granules.latitude.data - center[1])**2  
            argmin = (np.argmin(a))
            b = combined_granules.isel(longitude=neig[argmin,:], latitude=neig[argmin,:], AOD_550_Dark_Target_Deep_Blue_Combined=neig[argmin,:])
            mask = ((b.longitude.data >= XX[j,i+1]) & (b.longitude.data <= XX[j,i+1]+res) & (b.latitude.data >= YY[j, i+1]) & (b.latitude.data <= YY[j+1,i+1]+res))
            
            for modis_var in variables :
                out_modis_dsd[modis_var][j,i+1] = np.nanmean(b[modis_var].data[mask])
          #at the borders
          if (j == YY.shape[0]-2):
            center = ((XX[j+1,i] + XX[j+1,i+1])/2,(YY[j+1,i]+ YY[j+1,i]+res)/2)
            a  =(combined_granules.longitude.data - center[0])**2 + (combined_granules.latitude.data - center[1])**2  
            argmin = (np.argmin(a))
            b = combined_granules.isel(longitude=neig[argmin,:], latitude=neig[argmin,:], AOD_550_Dark_Target_Deep_Blue_Combined=neig[argmin,:])
            mask = ((b.longitude.data >= XX[j+1,i]) & (b.longitude.data <= XX[j+1,i+1]+res) & (b.latitude.data >= YY[j+1, i]) & (b.latitude.data <= YY[j+1,i]+res))
            
            for modis_var in variables :
                out_modis_dsd[modis_var][j+1,i] = np.nanmean(b[modis_var].data[mask])






  for modis_var in variables :
    out_modis_ds[modis_var] = (['south_north', 'west_east'], out_modis_dsd[modis_var])

  # Bilinear interp
  #for modis_var in variables :
  #    out_modis_ds[modis_var] = (
  #        ["south_north", "west_east"],
  #        griddata((combined_granules['longitude'], combined_granules['latitude']), combined_granules[modis_var], (XX, YY), method="linear"),
  #    )
  #    #out_modis_ds[modis_var] = nan_outside(out_modis_ds[modis_var], out_modis_ds['lon'], out_modis_ds['lat'], XX, YY, res/3)
  #    #out_modis_ds[modis_var] = nan_outside(out_modis_ds[modis_var], combined_granules['longitude'], combined_granules['latitude'], XX, YY, res)
  #    out_modis_ds[modis_var] = nan_outside(out_modis_ds[modis_var], combined_granules['longitude'], combined_granules['latitude'], XX, YY, res/3)

  out_modis_ds.coords['lon'] = (['south_north', 'west_east'],XX)
  out_modis_ds.coords['lat'] = ([ 'south_north', 'west_east',],YY)
  if which_modis == 'BOTH' :
    modis_root = f"./data/{region}/MODIS/" # for saving
  out_modis_ds.to_netcdf(modis_root + f'/regrided_MODIS_{which_modis}_AOD_550_{date}_{res}.nc')

if __name__ == "__main__" :
  if len(sys.argv) < 5 :
    print("args : start_day end_day resolution nprocessors which_modis(AQUA|TERRA|BOTH) region(AUSTRALIA|AFRICA|ASIA)")
    sys.exit()

  #res=0.2 #spatial resolution in degrees
  #hour=12 # chimere


  #dates = ['20191222']
  #for date in dates :
  print(sys.argv)

  d1 = sys.argv[1]
  d2 = sys.argv[2]
  res = float(sys.argv[3])
  procs = int(sys.argv[4])
  which_modis = sys.argv[5]
  region = sys.argv[6]

  dates  = count_days(d1, d2, '%Y%m%d')
  #dates  = count_days('20191201', '20191231', '%Y%m%d')

  if region == 'AUSTRALIA' :
    Z, Y, X  = grid_data(
        np.arange(13), 
        np.array([-39.0,-10.0]),
        np.array([110.0, 155.0]),
        res,
    )
  elif region == "AFRICA" :
    Z, Y, X  = grid_data(
        np.arange(8),
        np.array([-6.5, 38.05]),
        np.array([-19.0,52.55]),
        res,
    )
  elif region == "ASIA" :
    Z, Y, X  = grid_data(
        np.arange(8),
        np.array([20.0, 49.5]),
        np.array([100.0,145.0]),
        res,
    )
  else :
    raise('Unrecognized region (AUSTRALIA|AFRICA|ASIA)')
    sys.exit()

  XX, YY = X[0 ,:,:], Y[0,:,:]


  from functools import partial
  do_date1 = partial(do_date, res=res, which_modis=which_modis, region=region)
  with Pool(procs) as p :
    p.map(do_date1, dates )







  #import matplotlib.pyplot as plt
  #import cartopy.crs as ccrs
  #from aeros5py import cartopyaxes
  #ax = plt.subplot(projection=ccrs.PlateCarree())
  #im = plt.scatter(combined_granules['longitude'], combined_granules['latitude'], 0.5,  combined_granules['dtarget_dblue_aod55'], cmap='jet', 
  #     vmin=0, vmax=1)
  #cbar = plt.colorbar(im, ax=ax)
  #cbar.ax.tick_params(labelsize=20)
  ##ax.axes.set_title(f"MODIS {date} arccos(" + var + ")", fontsize=20)

  #ax.set_xticks(np.arange(110.0, 155.0, step=5));
  #ax.set_yticks(np.arange(-39.0,-10.0, step=5));
  #ax.set_xlim( 110.0, 155.0 )
  #ax.set_ylim( -39.0,-10.0 )
  #ax.axes.grid(b=True, which='both')
  #ax.coastlines()

  #plt.savefig('before_regrid')
  #plt.close('all')
