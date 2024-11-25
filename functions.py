import numpy as np
def grid_data(z, x, y, res):
    return np.mgrid[ z.min():z.max():1, x.min():x.max()+res:res, y.min():y.max()+res:res]

def nan_outside(mesh, lon0, lat0, X, Y, res) :
  """ Removes zeros outside the domain after resampling.
  https://stackoverflow.com/questions/30655749/how-to-set-a-maximum-distance-between-points-for-interpolation-when-using-scipy """
  THRESHOLD = res
  from scipy.interpolate.interpnd import _ndim_coords_from_arrays
  from scipy.spatial import cKDTree
  # Construct kd-tree, functionality copied from scipy.interpolate
  tree = cKDTree(np.array([lon0.flatten(), lat0.flatten()]).T)
  xi = _ndim_coords_from_arrays((X,Y), ndim=2)
  dists, indexes = tree.query(xi)
  # Copy original result but mask missing values with NaNs
  try : # for chiemre
        mesh = mesh.where(dists < THRESHOLD, np.nan)
  except : # for aeros5p
        mesh.data[dists > THRESHOLD] = np.nan
  return mesh

def nan_outside2(mesh, lon0, lat0, X, Y, res) :
  """ Removes zeros outside the domain after resampling."""
  from sklearn.neighbors import NearestNeighbors
  nbrs = NearestNeighbors(n_neighbors=2).fit( np.array([np.asarray(lon0).flatten(), np.asarray(lat0).flatten()]).T )
  dists, indexes = nbrs.kneighbors( np.array([np.asarray(lon0).flatten(), np.asarray(lat0).flatten()]).T)
  print(dists.min())
  mesh = mesh.where(dists[:,1].reshape(mesh.shape) < res, np.nan)
  return mesh


def plot_maxmin_points(ax, lon, lat, data, extrema, nsize, symbol, color='k',
                       plotValue=True, transform=None):
    """
    This function will find and plot relative maximum and minimum for a 2D grid. The function
    can be used to plot an H for maximum values (e.g., High pressure) and an L for minimum
    values (e.g., low pressue). It is best to used filetered data to obtain  a synoptic scale
    max/min value. The symbol text can be set to a string value and optionally the color of the
    symbol and any plotted value can be set with the parameter color
    lon = plotting longitude values (2D)
    lat = plotting latitude values (2D)
    data = 2D data that you wish to plot the max/min symbol placement
    extrema = Either a value of max for Maximum Values or min for Minimum Values
    nsize = Size of the grid box to filter the max and min values to plot a reasonable number
    symbol = String to be placed at location of max/min value
    color = String matplotlib colorname to plot the symbol (and numerica value, if plotted)
    plot_value = Boolean (True/False) of whether to plot the numeric value of max/min point
    The max/min symbol will be plotted on the current axes within the bounding frame
    (e.g., clip_on=True)
    https://unidata.github.io/python-gallery/examples/HILO_Symbol_Plot.html
    """
    from scipy.ndimage.filters import maximum_filter, minimum_filter

    if (extrema == 'max'):
        data_ext = maximum_filter(data, nsize, mode='nearest')
    elif (extrema == 'min'):
        data_ext = minimum_filter(data, nsize, mode='nearest')
    else:
        raise ValueError('Value for hilo must be either max or min')

    mxy, mxx = np.where(data_ext == data)

    for i in range(len(mxy)):
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]], symbol, color=color, size=7,
                clip_on=True, horizontalalignment='center', verticalalignment='center',
                transform=transform)
        ax.text(lon[mxy[i], mxx[i]], lat[mxy[i], mxx[i]],
                '\n' + str(np.int(data[mxy[i], mxx[i]])),
                color=color, size=7, clip_on=True, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=transform)

def load_inputs_only(date ,plot_corr=False, threshold=None) :
  """
  ### 2d data preparation
  """
  ch_ds, _, _ = load_datasets(date)

  # %%
  #reading dataset


  ch_ds = ch_ds.drop_dims(["bottom_top", "wavelength_dim"])

  # %%
  n = ch_ds.pblh.isnull()
  nanmask = n
  ch_ds = ch_ds.where(~nanmask)


  ch_df = ch_ds.to_dataframe().dropna()

  if plot_corr == True :
    plot_correlation_matrix(ch_df, f"CHIMERE correlation matrix 2D fields for {date}" )

  # %%
  input_layer = []

  for v in variables["var_2d_ch"] :
      input_layer.append(ch_df[v])
  input_layer = np.asarray(input_layer).T


  # %%
  """
  ### 3d data preparation
  """

  # %%
  #reading dataset
  ch_ds, _, _ = load_datasets(date)

  ch_ds = ch_ds.drop_dims(["wavelength_dim"])

  # %%
  for v in variables["var_2d_ch"] :
      ch_ds = ch_ds.drop_vars(v)

  # %%
  ch_ds = ch_ds.where(~nanmask)
  #else :
  #    ch_ds = ch_ds.where(~n)

  ch_df = ch_ds.to_dataframe().dropna()

  if plot_corr == True :
    plot_correlation_matrix(ch_df, f"CHIMERE correlation matrix 3D fields for {date}" )

  # %%
  input_layer3 = []
  for v in variables["var_3d_ch"] :
      input_layer3.append(ch_df[v])

  input_layer3 = np.asarray(input_layer3)

  # %%
  input_layer.shape

  # %%
  input_layer = np.append( input_layer, input_layer3.reshape(-1,input_layer.shape[0]).T, axis=1)

  if threshold != None :
    input_layer1 = input_layer[input_layer[:,2] < threshold[1]]
    input_layer = input_layer1[input_layer1[:,2] > threshold[0]]

  return input_layer


def set_metrics_str2(ax1, x, y, **kwargs):
    # y is the reference
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr

    masq = np.isnan(x) | np.isnan(y)
    #rsq = r2_score(x[~np.isnan(x)].reshape(-1, 1), y[~np.isnan(y)])
    rsq = r2_score(x[~masq], y[~masq])

    pearsonr = pearsonr(x[~masq], y[~masq])
    rmse = np.sqrt(mean_squared_error(x[~masq], y[~masq]))
    mae = mean_absolute_error(x[~masq], y[~masq])
    mb = np.mean(y[~masq]-x[~masq])
    #spearmanr = scipy.stats.spearmanr(x[~np.isnan(x)], y[~np.isnan(y)])
    #kendalltau = scipy.stats.kendalltau(x[~np.isnan(x)], y[~np.isnan(y)])
    textstr = '\n'.join((
        #f"$R^2$ = {np.round(rsq,2)}",
        f'r = {np.round(pearsonr,2)[0]}',
        f'RMSE = {np.round(rmse,2)}',
        f"MAE = {np.round(mae,2)}",
        #f"MB = {np.round(mb,2)}",
        )
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #ax1.text(0.55, 0.55, textstr, transform=ax1.transAxes, fontsize=fontsize,
    ax1.text(0.05, 0.80, textstr, transform=ax1.transAxes, **kwargs,

              verticalalignment='top', bbox=props)
    return ax1, {"rsq":rsq, "pearsonr":pearsonr[0], "rmse":rmse, "mae":mae, "mb":mb}


def set_metrics_str(ax1, x, y, **kwargs):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr


    masq = np.isnan(x) | np.isnan(y)
    #rsq = r2_score(x[~np.isnan(x)].reshape(-1, 1), y[~np.isnan(y)])
    rsq = r2_score(x[~masq].reshape(-1, 1), y[~masq])

    pearsonr = pearsonr(x[~masq], y[~masq])
    rmse = np.sqrt(mean_squared_error(x[~masq], y[~masq]))
    mae = mean_absolute_error(x[~masq], y[~masq])
    mb = np.mean(y[~masq]-x[~masq])
    #spearmanr = scipy.stats.spearmanr(x[~np.isnan(x)], y[~np.isnan(y)])
    #kendalltau = scipy.stats.kendalltau(x[~np.isnan(x)], y[~np.isnan(y)])
    textstr = '\n'.join((
        #f"$R^2$ = {np.round(rsq,2)}",
        f'r = {np.round(pearsonr,2)[0]}',
        f'RMSE = {np.round(rmse,2)}',
        f"MAE = {np.round(mae,2)}",
        #f"MB = {np.round(mb,2)}",
        )
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    #ax1.text(0.55, 0.55, textstr, transform=ax1.transAxes, fontsize=fontsize,
    #ax1.text(0.1, 0.90, textstr, transform=ax1.transAxes, **kwargs,
    ax1.text(0.55, 0.25, textstr, transform=ax1.transAxes, **kwargs,


              verticalalignment='top', bbox=props)
    #return ax1, {"rsq":rsq, "pearsonr":pearsonr[0], "rmse":rmse, "mae":mae, "mb":mb}
    return ax1, {"pearsonr":pearsonr[0], "rmse":rmse, "mae":mae}
def scatter_plot1(ax, x, y, z, vmin=0, vmax=5, cmap="YlOrBr", title=None, norm=None, **kwargs): # cmap="YlOrBr"
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, PowerNorm
    if norm in ['lin', None] :
        im = ax.axes.scatter(x, y, 1.5, z, marker='s', vmin=vmin, vmax=vmax, cmap=cmap)
    elif norm == 'sqrt' :
        im = ax.axes.scatter(x, y, 1.5, z, marker='s', cmap=cmap, norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=0.5))
    elif norm == 'log' :
        im = ax.axes.scatter(x, y, 1.5, z, marker='s', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))

    cbar = plt.colorbar(im, ax=ax, fraction=0.06)
    cbar.ax.tick_params(labelsize=10)

    ax.axes.set_title(title, fontsize=20)
    ax.coastlines()

def construct_altitudes(ds, levels=12) :
    hlay = np.empty(( ds['lon'].data.shape[0], ds['lon'].data.shape[1],12))
    for l in range(levels) :
        hlay[:,:,l] = l
    return hlay
def plot_correlation_matrix(df, title, method="pearson", **kwargs):
  import matplotlib.pyplot as plt
  #Correlation table
  #figsize=(30,15)
  figsize=(10,5)
  fig = plt.figure(figsize=figsize)
  plt.pcolormesh(df.dropna().corr(method=method), cmap='jet', **kwargs)
  plt.xticks(ticks=np.arange(0.5, len(list(df.columns)), 1), labels=list(df.columns))
  plt.yticks(ticks=np.arange(0.5, len(list(df.columns)), 1), labels=list(df.columns))
  plt.colorbar()
  plt.title(title)


def load_datasets(do_aeros5p=False, do_modis=False, do_fire=False, do_viirs=False, do_chimere=False,
                  common_na_mask=False, viirs_class=None, **kwargs) :
    """"Returns xarray of CHIMERE and/or AERO3D-S5P"""
    import xarray as xr
    import os
    from glob import glob
    #root_ch = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/CHIMERE/{kwargs['region']}/{kwargs['hour']}h/" # regrided
    #root_ch = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/CHIMERE/{kwargs['region']}/fewer/" # regrided
    #root_ch = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/CHIMERE/{kwargs['region']}/" # original
    #root_ch = f"/media/disk1/FAROUK2/" # sub local second simulations
    #root_ch = f"./prep/fewer/{kwargs['hour']}/{kwargs['res']}/" # sub local second simulations
    #root_ch = f"/media/disk1/{kwargs['region']}/CHIMERE/" # original local
    root_ch = f"./data//{kwargs['region']}/{kwargs['hour']}/{kwargs['res']}/" # updated with data given by Farouk Dec 22
    root_fp = "/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/MODIS/AUSTRALIA/FIRE/"

    root_ae = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/AUSTRALIA_AI/OUTPUTS/{kwargs['date']}/AEROS5P_MAX_v{kwargs['version']}_australia_big3/"
    root_modis = f"./data/{kwargs['region']}/MODIS/" # updated to the path with data given by Farouk Dec 22
    #root_modis = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/MODIS/{kwargs['region']}/{kwargs['which_inst']}/{kwargs['res']}/"
    #root_modis = f"/media/disk1/MODIS/{kwargs['which_inst']}/{kwargs['res']}/"
    root_viirs = f"/home/farouk/cluster/DATA/SPECAT_2/flemmouchi/VIIRS/AUSTRALIA/AOD/{kwargs['res']}/"
    masks_list = []
    ds = {}


    if do_aeros5p == True :
        aeros5p_filename = f"grided_log_a_AOD_vlidort_all_{kwargs['res']}.nc"
        ds['ae_ds'] = xr.open_dataset(root_ae + aeros5p_filename)
        #ds['ae_ds']['hlay'] = ( "south_north", "west_east", "bottom_top") , construct_altitudes(ds['ae_ds'], levels=12)

        masks_list.append(ds['ae_ds'].s5p_cloud_fraction.isnull())
    if do_chimere  == True:
        #chimere_filename = os.path.basename(glob(f"{root_ch}/regrided_{kwargs['hour']}h_{kwargs['res']}_out.{kwargs['date']}00_2*")[0]) # regrid
        chimere_filename = os.path.basename(glob(f"{root_ch}/out.{kwargs['date']}00_2*")[0]) # original

        ds['ch_ds'] = xr.open_dataset(root_ch + chimere_filename, engine="netcdf4")
        try :
          masks_list.append(ds['ch_ds'].pblh.isnull())
        except :
          masks_list.append(ds['ch_ds'].hght.isnull())
    if do_fire == True:
        fp_filename = f"regrided_MODIS_FIRE_AOD_550_{kwargs['date']}_{kwargs['res']}.nc"
        ds['fp_ds'] = xr.open_dataset(root_fp + fp_filename)
        masks_list.append(ds['fp_ds'].FP_power.isnull())
    if do_modis == True:
        #print(kwargs['which_inst'])
        modis_filename = f"regrided_MODIS_{kwargs['which_inst']}_AOD_550_{kwargs['date']}_{kwargs['res']}.nc"
        ds['modis_ds'] = xr.open_dataset(root_modis + modis_filename)
        masks_list.append(ds['modis_ds'].AOD_550_Dark_Target_Deep_Blue_Combined.isnull())
    if do_viirs == True :
        viirs_filename = f"regrided_VIIRS_AOD_{kwargs['date']}_{kwargs['res']}.nc"
        ds['viirs_ds'] = xr.open_dataset(root_viirs + viirs_filename)
        masks_list.append(ds['viirs_ds'].Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate.isnull())
        #classes = {0:"dust", 1:"smoke", 2:"high_altitude_smoke", 3:"pyrocumulonimbus_clouds", 4:"non-smoke_fine_mode", 5:"mixed", 6:"background"}
        if viirs_class in range(7) :
            masks_list.append(~(ds['viirs_ds'].Aerosol_Type_Land_Ocean.data == viirs_class))


    nanmask = masks_list[0]
    for m in masks_list[1:] :
      nanmask = m | nanmask
    if common_na_mask == False :
        nanmask = nanmask * 0
    if do_aeros5p == True:
        ds['ae_ds'] = ds['ae_ds'].where(~nanmask)
    if do_chimere == True:
        ds['ch_ds'] = ds['ch_ds'].where(~nanmask)
    if do_modis == True :
        ds['modis_ds'] = ds['modis_ds'].where(~nanmask)
    if do_viirs == True :
        ds['viirs_ds'] = ds['viirs_ds'].where(~nanmask)
    if do_fire == True :
        ds['fp_ds'] = ds['fp_ds'].where(~nanmask)
    if len(ds.keys()) > 1 :
        return ds.values()
    else :
        return ds[list(ds)[0]]

def load_input_output2(plot_corr=False, **kwargs) : #, threshold=None, **kwargs) : # using original chimere data (include 24h and 20 levels)
  from aeros5py.post_proc import convert_aod_wv
  import pandas as pd
  from datetime import datetime
  """
  ### 2d data preparation
  """
  ch_ds, modis_ds = load_datasets(**kwargs)
  optdaero = ch_ds.optdaero
   
  alpha =  -np.log(optdaero[:,2,:,:]/optdaero[:,3,:,:]) / np.log(400/600) 
  aod_550_mat = convert_aod_wv(optdaero[:,2,:,:], 400, 550, alpha)
   
  aod_550 = aod_550_mat.to_dataframe().dropna()["optdaero"]

  # %%
  #reading dataset

  ch_ds = ch_ds.drop_dims(["bottom_top", "wavelength_dim"])
  
  # %%
  modis_df = modis_ds.AOD_550_Dark_Target_Deep_Blue_Combined.to_dataframe().dropna()
  ch_df = ch_ds.to_dataframe().dropna()

  # %%

  reference_layer = []
  for v in kwargs["var_2d_modis"] :
    reference_layer.append(modis_df[v].values)
  reference_layer = np.asarray(reference_layer).T
  
  # %%
  
  input_layer = []
  input_layer.append(ch_df["lon"][:].values.flatten()) #1pm NEW
  input_layer.append(ch_df["lat"][:].values.flatten()) #1pm NEW
  #if "aod_550" in kwargs["var_2d_ch"]:
  input_layer.append(aod_550.values.flatten()) #1pm NEW
  #datee = datetime.strptime(kwargs['date'], "%Y%m%d")
  #input_layer.append(np.repeat(kwargs['date'], ch_df["lon"].size)) # date

  for v in kwargs["var_2d_ch"] :
  #  if v == "aod_550" : continue
    input_layer.append(ch_df[v].values) #1pm NEW
  input_layer = np.asarray(input_layer).T
  #print(input_layer.shape)
  if plot_corr == True :
    ch_df['aod_550'] = aod_550
    plot_correlation_matrix(ch_df, f"CHIMERE correlation matrix 2D fields for {kwargs['date']}" )
  #plot_correlation_matrix(ae_df, "AERO3D S5P correlation matrix 2D fields" )
  
  # %%
  """
  ### 3d data preparation
  """

  # %%
  #reading dataset
  ch_ds, _ = load_datasets(**kwargs)
    
  #ch_ds = ch_ds.drop_dims(["wavelength_dim"]) # commented to try with true vertical res

  # %%
  for v in ch_ds.var() :
      if v not in kwargs["var_3d_ch"]:
        ch_ds = ch_ds.drop_vars(v)

  ch_df = ch_ds.to_dataframe().dropna()

  if plot_corr == True :
    plot_correlation_matrix(ch_df, f"CHIMERE correlation matrix 3D fields for {kwargs['date']}" )

  # %%
  input_layer3 = []
  for v in kwargs["var_3d_ch"] :
      #print(v, end=", ")
      input_layer3.append(ch_df[v].values)
  input_layer3 = np.asarray(input_layer3).reshape(-1,input_layer.shape[0]).T
  
  # %%
  #print(input_layer.shape, input_layer3.shape)
  input_layer = np.append( input_layer, input_layer3, axis=1)

  #columns = ["lon", "lat", "date"] + kwargs["var_2d_ch"] 
  columns = ["lon", "lat", "aod_550"] + kwargs["var_2d_ch"] 
  for v3d in kwargs["var_3d_ch"] :
      for i in range(4) :
          columns.append(v3d + '_' + str(i))
            
  if kwargs['aod_threshold'] != None :
    input_layer1 = input_layer[input_layer[:,2] < kwargs['aod_threshold'][1]]
    input_layer2 = input_layer1[input_layer1[:,2] > kwargs['aod_threshold'][0]]

    reference_layer1 = reference_layer[input_layer[:,2] < kwargs['aod_threshold'][1]]
    reference_layer2 = reference_layer1[input_layer1[:,2] > kwargs['aod_threshold'][0]]


    return pd.DataFrame(input_layer2, columns=columns), pd.DataFrame(reference_layer2, columns=kwargs["var_2d_modis"])
  else :
    return pd.DataFrame(input_layer, columns=columns), pd.DataFrame(reference_layer, columns=kwargs["var_2d_modis"])

def plot_performance_panels(date=None, vmin=0, vmax=2, **config) :
  import numpy as np
  import xarray as xr
  import cartopy.crs as ccrs
  import matplotlib.pyplot as plt
  which_inst = config['which_modis']
  figpath = config['figpath']
  norm='sqrt'
  chim_ds = xr.open_dataset(f"{config['output_path']}/out.{date}00_01.nc")
  modis_ds = load_datasets(date=date, do_chimere=False, do_modis=True, **config)
  modis_aod = modis_ds.AOD_550_Dark_Target_Deep_Blue_Combined[37:,:]

  fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10,10), subplot_kw={'projection': ccrs.PlateCarree()}, constrained_layout=True)

  scatter_plot1(axes[0,0], chim_ds.lon, chim_ds.lat, chim_ds['optdaero_550.0'], vmin=0, vmax=vmax, title="CHIMERE raw.out", cbar_norm=norm)
  scatter_plot1(axes[1,0], chim_ds.lon, chim_ds.lat, chim_ds['optdaero_550.0_corr'], vmin=0, vmax=vmax, title="CHIMERE corrected.out", cbar_norm=norm)
  scatter_plot1(axes[2,0], chim_ds.lon, chim_ds.lat, modis_aod, vmin=0, vmax=vmax, title=f"MODIS {which_inst}", cbar_norm=norm)

  #FB
  #scatter_plot1(axes[2,1], chim_ds.lon, chim_ds.lat, 2*(prediction[:,j] - chim_ds['optdaero_550.0'])/(chim_ds['optdaero_550.0'] + prediction[:,j]) , vmin=-2, vmax=2, title="FB corrected vs raw", cmap="bwr")
  diff = chim_ds['optdaero_550.0_corr']-chim_ds['optdaero_550.0']
  scatter_plot1(axes[2,1], chim_ds.lon, chim_ds.lat, diff.data.flatten(), cmap="bwr", vmin=-0.5, vmax=0.5, title="AOD DIFF corrected-raw")

  fb = 2*(chim_ds['optdaero_550.0_corr']-modis_aod)/(modis_aod + chim_ds['optdaero_550.0_corr'])
  scatter_plot1(axes[1,1], chim_ds.lon, chim_ds.lat, fb, vmin=-2, vmax=2, title=f"FB corrected vs {which_inst}", cmap="bwr")
  fb = 2*(chim_ds['optdaero_550.0'] - modis_aod)/(chim_ds['optdaero_550.0'] + modis_aod)
  scatter_plot1(axes[0,1], chim_ds.lon, chim_ds.lat, fb, vmin=-2, vmax=2, title=f"FB raw vs {which_inst}", cmap="bwr")
  if date in config['train_dates'] :
      plt.suptitle(f"{config['model_type']} {date[0:4]}-{date[4:6]}-{date[6:8]}* test:{config['test_n']}", fontsize=20)
  else :
      plt.suptitle(f"{config['model_type']} {date[0:4]}-{date[4:6]}-{date[6:8]} test:{config['test_n']}", fontsize=20)

  axes[2,1], metrics = set_metrics_str2(axes[2,1], chim_ds['optdaero_550.0'], chim_ds['optdaero_550.0_corr'])
  axes[0,1], metrics0 = set_metrics_str2(axes[0,1], chim_ds['optdaero_550.0'], modis_aod)
  axes[1,1], metrics = set_metrics_str2(axes[1,1], modis_aod, chim_ds['optdaero_550.0_corr'])

  plt.savefig(f"{figpath}/result_"+ str(date) + ".jpeg"); plt.close()
  plt.close()
  return metrics, metrics0 

def plot_testing_dates(test_dates, config={}) :
    import pandas as pd
    import numpy as np
    from time import time
    log_reference = pd.DataFrame()
    log_prediction = pd.DataFrame()
    inference_cost = np.empty(0)

    for date in test_dates :
    #from multiprocessing import Pool
    #with Pool(2) as p :
    #     p.map(plot_date, test_dates)
        start = time()
        metrics, metrics0 =  plot_performance_panels(date, **config)
        end = time()
        inference_cost = np.append(inference_cost, end-start)
        log_reference   = log_reference.append(metrics0, ignore_index=True)
        log_prediction = log_prediction.append(metrics, ignore_index=True)
    return log_reference, log_prediction, inference_cost
