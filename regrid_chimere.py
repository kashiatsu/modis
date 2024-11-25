# %matplotlib inline
#import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
#from aeros5py import cartopyaxes
import numpy as np
import xarray as xr
from scipy.interpolate import LinearNDInterpolator #NearestNDInterpolator
#from scipy.interpolate import griddata
import datetime as dt
from glob import glob
from aeros5py.utils import count_days
from functions import *
from multiprocessing import Pool 
import sys, os

#import pkg_resources.py2_warn, cftime

def do_date(date, res, hour, wv, region) :
  print(date)
  root_ch = f'./data/{region}/13/'
  variables = dict()

  variables["2d"] = ["albedo", "tem2", "u10m", "usta", "v10m", "wsta"]
  if region == "AUSTRALIA" :
    chimere_filename = f"out.{date}00_24_AUST25.nc.sub"
    variables["2d"] = variables["2d"] + ["pblh", "ws10m"]
    hmax=15 #max altitude index chiemre
    Z, Y, X  = grid_data(
        np.arange(13), 
        np.array([-39.0,-10.0]),
        np.array([110.0, 155.0]),
        res,
    )
  elif region == "AFRICA" :
    hmax=20 #max altitude index chiemre
    variables["2d"] = variables["2d"] + ["hght", "w10m"]
    chimere_filename = os.path.basename(glob(f"{root_ch}/out.{date}*")[0])
    Z, Y, X  = grid_data(
        np.array([0,5]),
        np.array([-6.5, 38.05]),
        np.array([-19.0,52.55]),
        res,
    )
  elif region == "ASIA" :
    hmax=4 #max altitude index chimere of input files (already reduced in vertical levels)
    variables["2d"] = variables["2d"] + ["hght", "w10m"]
    chimere_filename = os.path.basename(glob(f"{root_ch}/out.{date}*")[0])
    # Target resolution X (lon), Y (lat), Z (wavelength/altitude)
    Z0, Y0, X0  = grid_data(
        np.array([0,5]),
        np.array([20.0, 49.5]),
        np.array([100.0,145.0]),
        res,
    )
    Z, Y, X = Z0[0:4,:,:], Y0[0:4,:,:], X0[0:4,:,:]

  wavelength = 200, 300, 400, 600, 999 
  hmin=0 #min altitude index chimere
  #projection=ccrs.PlateCarree()
  #interp_routine = NearestNDInterpolator
  interp_routine = LinearNDInterpolator
  cdata = xr.open_dataset(root_ch + chimere_filename, engine='netcdf4')
  cdata = cdata.assign_coords({"lon":cdata.lon, "lat":cdata.lat})
  #Z[0,:,:] = np.ceil(cdata.hlay[hour,0,:,:]) #so that levels don't start from 0.
  tmp_alt0 = 0.07 #np.ceil(cdata.hlay[hour,0,:,:].max()) / 1000

  Z[0,:,:] = tmp_alt0 #np.max(tmp_alt0) #np.where( tmp_alt0 > 0.5, tmp_alt0, 0.5 )  #so that levels don't start from 0.
  #print(Z[:,0,0])

  ch_ds = xr.Dataset()
  
  #print(cdata.wavelength.values)
  # XX, YY 2D lon lat target
  XX, YY = X[0,:,:], Y[0,:,:]
  #Use same aeros5p grid
  ch_ds.coords['lon'] = (['south_north', 'west_east'],XX)
  ch_ds.coords['lat'] = (['south_north', 'west_east'],YY)
  ch_ds['wavelength'] = (['wavelength_dim'], cdata.wavelength.values)
  
#  variables["2d"] = []  #"aerr", "albedo", "atte", "copc", "flashnum", "lastrain", "lspc", "pblh", "slhf", "soim", "sreh", "sshf", "swrd", "tem2", "topc", "u10m", "usta", "v10m", "ws10m", "wsta" ]

#  variables["2d"] = [ "albedo", "copc", "lspc", "pblh", "slhf", "soim", "sreh", "sshf", "swrd", "tem2", "topc", "u10m", "usta", "v10m", "ws10m", "wsta" ]

#  variables["3d"] = ['APINEN', 'AeropH', 'AnA1D', 'AnA1DAQ', 'AnBmP', 'AnBmPAQ', 'BCARAQ', 'BPINEN', 'BiA1D', 'BiA1DAQ', 'BiBmP', 'BiBmPAQ', 'C2H4',
#                     'C2H6', 'C3H6', 'C5H8', 'CH3CHO', 'CH3COE', 'CH4', 'CO', 'CloudpH', 'DMS', 'DMSO', 'DUSTAQ', 'GLYOX', 'H2O2', 'H2SO4', 'H2SO4AQ',
#                     'HCHO', 'HCL', 'HCLAQ', 'HCNM', 'HNO3', 'HNO3AQ', 'HONO_AER', 'HONO_ANT', 'HONO_CHEM', 'HONO_FIRE', 'HONO_SURF', 'HOX', 'HUMULE',
#                     'ISOPA1', 'ISOPA1AQ', 'LIMONE', 'MGLYOX', 'MSA', 'NAAQ', 'NC4H10', 'NH3', 'NH3AQ', 'NO', 'NO2', 'NOX', 'NOY', 'O3', 'OCARAQ',
#                     'OCIMEN', 'OH', 'OX', 'OXYL', 'PAN', 'PM10', 'PM10ant', 'PM10bio', 'PM25', 'PM25ant', 'PM25bio', 'PPMAQ', 'ROOH', 'SO2', 'TERPEN',
#                     'TMB', 'TOL', 'TOTPAN', 'airm', 'chem_regime', 'cice', 'cliq', 'dpdd', 'dpdu', 'dpeu', 'flxd', 'flxu', 'kzzz',
#                     'pAnA1D', 'pAnBmP', 'pBCAR', 'pBiA1D', 'pBiBmP', 'pDUST', 'pH2SO4', 'pHCL', 'pHNO3', 'pISOPA1', 'pNA', 'pNH3', 'pOCAR', 'pPPM',
#                     'pWATER', 'pres', 'rain', 'relh', 'snow', 'sphu', 'temp', 'winm', 'winw', 'winz']

  #variables["3d"] = ['AeropH','BCARAQ', 'CH4', 'CO', 'DMS', 'DMSO', 'DUSTAQ', 'GLYOX', 'H2O2', 'H2SO4', 'H2SO4AQ',
  #                   'HCHO', 'HCL', 'HCLAQ', 'HCNM', 'HNO3', 'HNO3AQ', 'HONO_AER', 'HONO_ANT', 'HONO_CHEM', 'HONO_FIRE', 'HONO_SURF', 'HOX', 'HUMULE',
  #                   'NAAQ', 'NH3', 'NH3AQ', 'NO', 'NO2', 'NOX', 'NOY', 'O3', 'OCARAQ',
  #                   'OH', 'OX', 'PM10', 'PM10ant', 'PM10bio', 'PM25', 'PM25ant', 'PM25bio', 'PPMAQ', 'ROOH', 'SO2',
  #                   'TOL', 'airm', 'pBCAR', 'pDUST', 'pH2SO4', 'pHCL', 'pHNO3', 'pNA', 'pNH3', 'pOCAR', 'pPPM',
  #                   'pWATER', 'pres', 'rain', 'relh', 'snow', 'sphu', 'temp', 'winm', 'winw', 'winz']

#  variables["3d"] = ["O3","NO2","NO","PAN","HNO3","H2O2","H2SO4","HONO","HONO_CHEM","HONO_AER","HONO_SURF","HONO_ANT","HONO_FIRE","SO2","CO","OH",
#                    "CH4","C2H6","NC4H10","C2H4","C3H6","OXYL","C5H8","HCHO","CH3CHO","GLYOX","MGLYOX","CH3COE","NH3","APINEN","BPINEN","LIMONE",
#                    "TERPEN","OCIMEN","HUMULE","TOL","TMB","AnA1D","AnBmP","BiA1D","BiBmP","ISOPA1","PPMAQ","H2SO4AQ","HNO3AQ","NH3AQ","AnA1DAQ",
#                    "AnBmPAQ","BiA1DAQ","BiBmPAQ","ISOPA1AQ","HCL","HCLAQ","NAAQ","DUSTAQ","OCARAQ","BCARAQ","bBCAR","bDUST","bNA","bOCAR","bPPM",
#                    "bAnA1D","bAnBmP","bBiA1D","bBiBmP","bISOPA1","bH2SO4","bHCL","bHNO3","bNH3","bWATER","DMS","DMSO","MSA","TOTPAN","NOX","OX",
#                    "HOX","NOY","ROOH","HCNM","pBCAR","pDUST","pNA","pOCAR","pPPM","pAnA1D","pAnBmP","pBiA1D","pBiBmP","pISOPA1","pH2SO4","pHCL",
#                    "pHNO3","pNH3","pWATER","PM25","PM25bio","PM25ant","PM10","PM10bio","PM10ant","chem_regime","bnum","winz","winm","winw","temp",
#                    "sphu","airm","hlay","pres","relh","CloudpH","AeropH","kzzz","cliq","rain","snow","cice","thlay","dpeu","dped","dpdu","dpdd","flxu",
#                    "flxd","jO3","jNO2","dry_to_wet","wet_density","atb_mol_nad","atb_tot_nad","scat_ratio_nad","atb_mol_zen","atb_tot_zen","scat_ratio_zen"]

  #variables["3d"] = ["AeropH", "CH4", "CO", "CloudpH", "H2SO4", "H2SO4AQ", "HCHO", "HNO3", "HNO3AQ", "HONO_AER", "HONO_ANT", "HONO_CHEM", "HONO_FIRE", "HONO_SURF", 
  #                   "HOX", "HUMULE", "NAAQ", "NH3", "NH3AQ", "NO2", "NOX", "NOY", "O3", "OCARAQ", "OH", "OX", "PM10", "PM25", "PPMAQ", "ROOH", "SO2", "TOL", 
  #                   "airm",  "cice", "cliq", "dpdd", "dpdu", "dped", "dpeu", "flxd", "flxu", "kzzz", "pBCAR",
  #                   "pDUST", "pH2SO4", "pHCL", "pHNO3", "pNA", "pNH3", "pOCAR", "pPPM", "pWATER", "pres", "relh", "sphu",
  #                   "temp", "winm", "winw", "winz"]

  #variables["3d"] = ["CH4", "CO", "H2SO4", "H2SO4AQ", "HCHO", "HNO3", "HNO3AQ", "HONO_AER", "HONO_ANT", "HONO_CHEM", "HONO_FIRE", "HONO_SURF",
  #                   "HOX", "HUMULE", "NAAQ", "NH3", "NH3AQ", "NO2", "NOX", "NOY", "O3", "OCARAQ", "OH", "OX", "PM10", "PM25", "PPMAQ", "ROOH", "SO2", "TOL",
  #                   "airm",  "cice", "cliq", "dpdd", "dpdu", "dped", "dpeu", "flxd", "flxu", "kzzz", "pBCAR",
  #                   "pDUST", "pH2SO4", "pHCL", "pHNO3", "pNA", "pNH3", "pOCAR", "pPPM", "pWATER", "pres", "relh", "sphu",
  #                   "temp", "winm", "winw", "winz"]

  #variables["3d"] = [ "temp", "hlay", "pres", "PM25", "PM10", "pDUST", "pBCAR", "CO", "NO2", "O3", "pOCAR", "pres", "relh", "temp", "winm", "winw", "winz" ]

  # ASIA new files
  #variables["3d"] =  [ "PM10", "PM25", "O3", "SO2", "CO", "OH", "CH4", "NH3", "NOX", "OX", "HOX", "NOY", "pDUST", "pOCAR", "pNA", "pH2SO4", "pHNO3", "pNH3","pWATER", "airm", "relh", "temp", "winz", "winm", "winw", "sphu", "pres", "hlay"]

  # ASIA large files
  variables["3d"] =  [ "PM10", "PM25", "O3", "SO2", "CO", "OH", "CH4", "NH3", "NOX", "OX", "HOX", "NOY", "pDUST", "pOCAR", "pNA", "pH2SO4", "pHNO3", "pNH3","pWATER", "airm", "relh", "temp", "winz", "winm", "winw", "sphu", "pres", "hlay","BENZ","HCHO","C5H8","NO","NO2","H2SO4","pBCAR"]

  variables["2d"] =  ["hght", "albedo", "usta", "wsta", "copc", "lspc", "sshf", "swrd", "topc", "slhf", "soim", "sreh", "atte", "u10m", "tem2"] 
    
  variables["4d"] = ["bBCAR","bDUST","bNA","bOCAR","bH2SO4","bHNO3","bNH3","bWATER"]

  if region == "AUSTRALIA" or (region == "AFRICA" and res != 0.45) :
    for var2d in variables["2d"]:
        interpolator = interp_routine(
             (cdata.lon.data.flatten(), cdata.lat.data.flatten()),
             cdata[var2d].data[hour,:,:].flatten()
        )
        ch_ds[var2d] = xr.DataArray(data=interpolator(XX,YY), dims=["south_north", "west_east"])
        ch_ds[var2d] = nan_outside(ch_ds[var2d], cdata.lon.data, cdata.lat.data, XX, YY, res/2)

  elif region == "AFRICA" and res == 0.45 :
    for var2d in variables["2d"]:
      ch_ds[var2d] = cdata[var2d][hour,:]

  elif region == "ASIA" and res == 0.45 :
    for var2d in variables["2d"]:
        print(var2d, end=", ", flush=True)
        interpolator = interp_routine(
             (cdata.lon.data.flatten(), cdata.lat.data.flatten()),
             cdata[var2d].data[hour,:,:].flatten()
        )
        ch_ds[var2d] = xr.DataArray(data=interpolator(XX,YY), dims=["south_north", "west_east"])
        ch_ds[var2d] = nan_outside(ch_ds[var2d], cdata.lon.data, cdata.lat.data, XX, YY, res/2)

  alon = []; blat = []
  for i in range(hmax-hmin) :
    alon.append(cdata.lon.data)
    blat.append(cdata.lat.data)
  lon = np.array(alon) #, dtype=np.float64)
  lat = np.array(blat) #, dtype=np.float64)

  for var3d in variables["3d"] :
      print(var3d, end=", ", flush=True)
      interpolator = interp_routine(
          ( (cdata.hlay.data[hour,hmin:hmax,:,:].flatten()) / 1000,
          lon.flatten(), lat.flatten()),
          cdata[var3d].data[hour,hmin:hmax,:,:].flatten())

      #if region == "AFRICA" and res == 0.45 :
      #  tmp_array = Z * np.nan
      #  tmp_array[0,:,:] = cdata[var3d][hour,0,:,:]
      #  tmp_array[1:,:,:] = interpolator(Z[1:,:,:], X[1:,:,:], Y[1:,:,:])
      #  ch_ds[var3d] = (['bottom_top', 'south_north', 'west_east'], tmp_array)
      #else :
      #  ch_ds[var3d] = (['bottom_top', 'south_north', 'west_east'], interpolator(Z,X,Y))
      ch_ds[var3d] = (['bottom_top', 'south_north', 'west_east'], interpolator(Z,X,Y))
      ch_ds[var3d] = nan_outside(ch_ds[var3d], cdata.lon.data, cdata.lat.data, X, Y, res)

  ch_ds["Z"] = (['bottom_top', 'south_north', 'west_east'], Z)
  #for optical depth
  wavelength = 200, 300, 400, 600, 999 
  
  interpolator = interp_routine(
  (
      np.meshgrid(cdata.lon, cdata.wavelength.data[0:4])[1].reshape(4, cdata.lon.shape[0], cdata.lon.shape[1]).flatten(),
      lat.flatten(),
      lon.flatten(),
  ),
  cdata.optdaero.data[hour,0:4,:,:].flatten())
  
  optdaero = interpolator(
       np.meshgrid(XX[:,:], cdata.wavelength.data[0:5] )[1].reshape(5,XX.shape[0], XX.shape[1]),
       X0[0:5,:,:],
       Y0[0:5,:,:])
  
    
  optdaero_wv = interpolator(
      np.meshgrid(XX[:,:], wv )[1].reshape(XX.shape[0], XX.shape[1]),
      YY[:,:], 
      XX[:,:],
  )
  ch_ds[f'optdaero_{wv}'] =  (['south_north', 'west_east'], optdaero_wv)

  optdaero_wv1 = interpolator(
      np.meshgrid(XX[:,:], 200 )[1].reshape(XX.shape[0], XX.shape[1]),
      YY[:,:], 
      XX[:,:],
  )
  optdaero_wv2 = interpolator(
      np.meshgrid(XX[:,:], 300 )[1].reshape(XX.shape[0], XX.shape[1]),
      YY[:,:], 
      XX[:,:],
  )
  optdaero_wv3 = interpolator(
      np.meshgrid(XX[:,:], 400 )[1].reshape(XX.shape[0], XX.shape[1]),
      YY[:,:], 
      XX[:,:],
  )
  optdaero_wv4 = interpolator(
      np.meshgrid(XX[:,:], 600 )[1].reshape(XX.shape[0], XX.shape[1]),
      YY[:,:], 
      XX[:,:],
  )

  optdaero[0,:,:] = optdaero_wv1
  optdaero[1,:,:] = optdaero_wv2
  optdaero[2,:,:] = optdaero_wv3
  optdaero[3,:,:] = optdaero_wv4
  optdaero[4,:,:] = optdaero_wv4

  optdaero1 = np.zeros((1,X0.shape[0],X0.shape[1],X0.shape[2]))
  optdaero1[0,:,:,:] = optdaero
    
#  ch_ds['optdaero'] =  (['wavelength_dim','south_north', 'west_east'], optdaero)
  ch_ds['optdaero'] =  (['Time','wavelength_dim','south_north', 'west_east'], optdaero1)
  print('optdaero', end=", ", flush=True)

  mmd = 1.48347901906296e-08, 3.26470727918293e-08, 7.18467647497088e-08, 
        1.58113902537127e-07, 3.47962810171087e-07, 7.65765403379651e-07, 
        1.68522772941819e-06, 3.53553390593274e-06, 7.07106781186548e-06, 2e-05

  for var4d in variables["4d"] :
      print(var4d, end=", ", flush=True)
      interpolator4 = interp_routine(
          ( mmd,(cdata.hlay.data[hour,hmin:hmax,:,:].flatten()) / 1000,
          lon.flatten(), lat.flatten()),
          cdata[var4d].data[hour,hmin:hmax,:,:].flatten())

      ch_ds[var4d] = (['nbins','bottom_top', 'south_north', 'west_east'], interpolator4(mmd,Z,X,Y))
      ch_ds[var4d] = nan_outside(ch_ds[var3d], cdata.lon.data, cdata.lat.data, X, Y, res)

  ch_ds["Z"] = (['nbins','bottom_top','south_north', 'west_east'], Z)


  cdata.close()
  ch_ds.to_netcdf(f"{root_ch}/regrided_{hour}h_{res}_{chimere_filename}")

if __name__ == "__main__" :
  if len(sys.argv) < 6 :
    print("args : start_day end_day resolution hour nprocessors aod_wavelength region[AUSTRALIA/AFRICA/ASIA]")
    sys.exit()

  #res=0.2 #spatial resolution in degrees
  #hour=12 # chimere

  #for date in dates :
  print(sys.argv)

  d1 = sys.argv[1]
  d2 = sys.argv[2]
  res = float(sys.argv[3])
  hour = int(sys.argv[4])
  procs = int(sys.argv[5])
  wv = float(sys.argv[6])
  region = sys.argv[7]

  dates  = count_days(d1, d2, '%Y%m%d')

  from functools import partial
  do_date1 = partial(do_date, res=res, hour=hour, wv=wv, region=region)
  with Pool(procs) as p :
    p.map(do_date1, dates)
