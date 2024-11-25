#!/home/flemmouchi/.conda/envs/py36/bin/python3.6
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def plot_spectra(DICT1, DICT2=None, scanline=0, groundpixel=0, var='radiance', scale='normal', xmin=None, xmax=None):
  """Plot all bands spectra for a specific pixel.

    args:
        DICT1/DICT2 (dict): The keys should be the spectral bands.
        scanline (int): Along track pixel index (standard=0).
        groundpixel (int): Cross track pixel index (standard=0).
        var (str): radiance or irradiance
        scale (str): log (standard linear)
        xmin (int): xlim
        xmax (int): xlim

    return:
        plot of all bands for the pixel.

    example:
        plot_spectra(IRRADIANCE, scale='log')
        plot_spectra(RADIANCE, DICT2=IRRADIANCE, groundpixel=44) # to plot reflectance 
    """

  tmp = list(DICT1.keys())
  
  if var in DICT1[tmp[0]].keys() :
      for bd in DICT1.keys():
        if DICT2 != None : #reflectance
            if scale=='log':
                plt.plot(DICT1[bd]['nominal_wavelength'][0,0,:], np.log(DICT1[bd]['radiance'][0,scanline,groundpixel,:]/DICT2[bd]['irradiance'][0,0,groundpixel,:]), label=bd)
                var='reflectance'
            else :
                plt.plot(DICT1[bd]['nominal_wavelength'][0,0,:], DICT1[bd]['radiance'][0,scanline,groundpixel,:]/DICT2[bd]['irradiance'][0,0,groundpixel,:], label=bd)
                var='reflectance'
        else:
            if scale=='log':
                plt.plot(DICT1[bd]['nominal_wavelength'][0,groundpixel,:], np.log(DICT1[bd][var][0,scanline,groundpixel,:]), label=bd)
            else :
                plt.plot(DICT1[bd]['nominal_wavelength'][0,groundpixel,:], DICT1[bd][var][0,scanline,groundpixel,:], label=bd)
        
  else : #irradiance
    if DICT2 == None :
        plot_spectra(DICT1=DICT1, scanline=0, groundpixel=groundpixel, var='irradiance', scale=scale)
    else :
        plot_spectra(DICT1=DICT2, DICT2=DICT1, scanline=scanline, groundpixel=groundpixel, var='radiance', scale=scale) #reflectance
  plt.xlabel('Wavelength [nm]')
  plt.ylabel(f'{var} ua')
  plt.legend()
  plt.grid()
  plt.xlim(xmin,xmax)
  plt.title(f'{scale} scale {var} (scanline,groundpixel) ({scanline},{groundpixel})')
  plt.show()


def plot_isrf(ISRF, groundpixel=0, iwavelength=0):
    """Plot instrument spectral response function.

    args:
        ISRF (dict): -
        groundpixel (int): -
        iwavelength (int): index of wavelength

    return:
        plot of ISRF

    example:
        plot_isrf(ISRF, 5, 6)
    """
    for bd in ISRF.keys() :
      plt.plot(ISRF[bd]['delta_wavelength'][:], ISRF[bd]['isrf'][groundpixel,iwavelength,:], label=bd)
    plt.xlabel('dlambda [nm]')
    plt.ylabel('isrf')
    plt.xlim([-0.7,0.7])
    plt.ylim([-0.5,3])
    plt.legend()
    plt.grid()
    plt.title(f"ISRF at swath {groundpixel} {[ISRF[bd]['wavelength'][:].data[groundpixel,iwavelength] for bd in ISRF.keys()]}")
    plt.show()
    




def subscatter(i,j,k,x,y,n=None,c=None,vmin=None,vmax=None,title=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
  """args:
        i : nRows subplots.
        j : nColumns subplots.
        k : Index subplot.
        x : Latitude.
        y : Longitude.
        n : Scatter size.
        vmin : -
        vmax : -
        title : -"""
  projection=ccrs.PlateCarree()
  #lat1=-40.
  #lat2=-15.
  #lon1=120.
  #lon2=150.
  step=5
  ax=plt.subplot(i,j,k,projection=projection)
  #ax=plt.subplot(i,j,k)
  cmap=plt.cm.jet
  plt.scatter(x,y,n,c=c,cmap=cmap,vmin=vmin,vmax=vmax)
  plt.colorbar()
  ax.coastlines()
  #plt.xticks(np.arange(lon1,lon2, step=step));
  #plt.yticks(np.arange(lat1,lat2, step=step));
  #ax.set_xlim( lon_min=lon1, lon_max=lon2)
  #ax.set_ylim( lat_min=lat1, lat_max=lat2)
  plt.grid(b=True)
  plt.title(title)    


def full_frame(width=None, height=None):
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    figsize = None if width is None else (width, height)
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    return fig


def set_subaxis( ax, title, im,  lonlim=[110, 160], latlim=[-40,-10], step=5, last_one=None, plot_only=None, do_colorbar=True):
    ax.title.set_text(title)
    lon1=lonlim[0]-2; lon2=lonlim[1]+2 
    lat1=latlim[0]-2; lat2=latlim[1]+2 
    #lon1=115; lon2=150; lat1=-40; lat2=-15

    #lon1=115; lon2=160; lat1=-45; lat2=-10

    ax.coastlines()
    ax.set_xticks(np.arange(lon1,lon2, step=step));
    ax.set_yticks(np.arange(lat1,lat2, step=step));
    ax.set_xlim( xmin=lon1, xmax=lon2)
    ax.set_ylim( ymin=lat1, ymax=lat2)
    ax.axes.grid(b=True, which='both')
    if do_colorbar :
      plt.colorbar(im)
    if last_one is True :
      if plot_only == 'aod_lambdas' :
        print(last_one)
        print(plot_only)
        plt.colorbar(im)

    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    #ax.set_xticklabels(data.lon.data)
    #ax.set_yticklabels(data.lat.data)
    #ax.set_extent([data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()])



def subscatter2(i,j,k,x,y,n=None,c=None,vmin=None,vmax=None,title=None, trans_coo=None, last_one=None, plot_only=None):
    """args:
          i : nRows subplots.
          j : nColumns subplots.
          k : Index subplot.
          x : Latitude.
          y : Longitude.
          n : Scatter size."""
    projection=ccrs.PlateCarree()
    ax=plt.subplot(i,j,k,projection=projection)
    cmap=plt.cm.jet
    im = plt.scatter(x,y,n,c=c,cmap=cmap,vmin=vmin,vmax=vmax)
    if trans_coo is not None:
      set_subaxis(ax, title, im, [trans_coo[:,1].min(), trans_coo[:,1].max()], [trans_coo[:,2].min(), trans_coo[:,2].max()], 5, last_one=last_one, plot_only=plot_only)
    else :
      set_subaxis(ax, title, im, step=5, last_one=last_one, plot_only=plot_only)


def subpcolormesh(i,j,k,xi,yi,zi,vmin=None,vmax=None,title=None, alpha=None):
        from matplotlib.axes import Axes
        from cartopy.mpl.geoaxes import GeoAxes
        GeoAxes._pcolormesh_patched = Axes.pcolormesh

        ax=plt.subplot(i,j,k,projection=ccrs.PlateCarree())
        im = plt.pcolormesh(xi, yi, zi, cmap=plt.cm.jet,vmin=vmin, vmax=vmax, alpha=alpha)
        set_subaxis(ax, title, im)


def grid_data(x, y, z, res):
    from scipy.interpolate import griddata
    xi0 = np.arange(x.min(),x.max()+res,res)
    yi0 = np.arange(y.min(), y.max()+res,res)
    xi , yi = np.meshgrid(xi0, yi0)
    print(xi.shape)
    print(yi.shape)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    zi = griddata((x.flatten(), y.flatten()),z.flatten(),(xi,yi),method='cubic')
    return xi, yi,zi

