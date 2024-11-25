import numpy as np

def count_days(start, end, format) :
  """
  Arguments :
    start, end, format
  Return :
    list of days.
  """
  import datetime as dt  
  start_date =  dt.datetime.strptime(start, format)
  end_date = start_date
  dates = []
  print(f"Counting days...from {start} to {end}")
  while end_date.strftime(format) != end :
    dates.append(end_date.strftime(format))
    end_date = end_date + dt.timedelta(days=1)
  dates.append(end_date.strftime(format)) #last day
  return dates

def make_1d_mask(coo, lonlim, latlim):
  if 'lon' in coo :
    mask = ( (coo['lon'] < lonlim[1]) & (coo['lon'] > lonlim[0]) &
             (coo['lat'] < latlim[1]) & (coo['lat'] > latlim[0]) )
  else :
    mask = ( (coo['Subsatellite_Longitude'].data < lonlim[1]) & (coo['Subsatellite_Longitude'].data > lonlim[0]) &
             (coo['Subsatellite_Latitude'].data < latlim[1]) & (coo['Subsatellite_Latitude'].data > latlim[0]) )
  return mask



def colocate_generic(LR_lon, LR_lat, HR_longitude, HR_latitude, radius=0.2, lonlim=(110., 155.0), latlim=(-39.0,-10.0)) :
  """radius is in degrees. [default = 0.2]
     Return : LR indices nearest to HR
  """
  HR_DICT = dict()
  HR_DICT['lat'] = np.asarray(HR_latitude, dtype=np.float)
  HR_DICT['lon'] = np.asarray(HR_longitude, dtype=np.float)

  #mask = make_1d_mask(HR_DICT, [LR_lon.min(), LR_lon.max()],[LR_lat.min(), LR_lat.max()])
  mask = make_1d_mask(HR_DICT, lonlim,latlim)
  HR_DICT['lat'] = HR_DICT['lat'][mask]
  HR_DICT['lon'] = HR_DICT['lon'][mask]
  idx = HR_DICT['lon']*np.nan
  for j,HR_lat in enumerate(HR_DICT['lat']) :
    HR_lon  = HR_DICT['lon'][j]
    d = np.sqrt( (LR_lon - HR_lon)**2 + (LR_lat - HR_lat)**2 )
    if d.min() <= radius :
      idx[j] = d.argmin()
  idx = np.unique(idx) #remove ducplicates
  m = np.isnan(idx)
  idx = idx[~m]
  return idx.astype(int)


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    https://stackoverflow.com/questions/54962004/implement-max-mean-poolingwith-stride-with-numpy/54966908
    '''
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)

def do_pooling(lon, lat, cm, mode, pool_size, stride):
  import numpy as np
  lons = np.empty((int(cm.shape[0]/stride), int(cm.shape[1]/stride))) *np.nan
  lats = np.empty((int(cm.shape[0]/stride), int(cm.shape[1]/stride))) *np.nan
  cms = np.empty( (int(cm.shape[0]/stride), int(cm.shape[1]/stride))) *np.nan

  for iidx, i in enumerate(range(pool_size,cm.shape[0],stride)) : 
    for jidx, j in enumerate(range(pool_size,cm.shape[1],stride)): 
      lons[iidx,jidx] = lon[i-pool_size:i, j-pool_size:j].mean().mean() 
      lats[iidx,jidx] = lat[i-pool_size:i, j-pool_size:j].mean().mean()
      if mode == 'max':
        cms[iidx,jidx]  =  cm[i-pool_size:i, j-pool_size:j].max()
      elif mode == 'min':
        cms[iidx,jidx]  =  np.nanmin(cm[i-pool_size:i, j-pool_size:j])
      elif mode == 'average':
        cms[iidx,jidx]  =  cm[i-pool_size:i, j-pool_size:j].mean()

  lons = lons[~np.isnan(lons)]
  lats = lats[~np.isnan(lats)]
  cms = cms[~np.isnan(cms)]

  return lons, lats, cms

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
    return ax1, {"rsq":rsq, "pearsonr":pearsonr[0], "rmse":rmse, "mae":mae, "mb":mb}

  
#def set_metrics_str(x, y) :
#    from sklearn.metrics import r2_score
#    from scipy.stats import pearsonr
#    from sklearn.linear_model import LinearRegression
#    from sklearn.metrics import mean_absolute_error, mean_squared_error
#
#    #rsq = reg.score(x[~np.isnan(x)].reshape(-1, 1), y[~np.isnan(y)])
#    rsq = r2_score(x, y)
#    pearsonr = pearsonr(x[~np.isnan(x)], y[~np.isnan(y)])
#    rmse = mean_squared_error(x[~np.isnan(x)], y[~np.isnan(y)])
#    mae = mean_absolute_error(x[~np.isnan(x)], y[~np.isnan(y)])
#    #spearmanr = scipy.stats.spearmanr(x[~np.isnan(x)], y[~np.isnan(y)])
#    #kendalltau = scipy.stats.kendalltau(x[~np.isnan(x)], y[~np.isnan(y)])
#    textstr = '\n'.join((
#        f"$R^2$ = {np.round(rsq,2)}",
#        f'r = {np.round(pearsonr,2)[0]}',
#        f'RMSE = {np.round(rmse,2)}',
#        f"MAE = {np.round(mae,2)}",
#        )
#    )
#    return textstr, rsq, pearsonr, rmse, mae
