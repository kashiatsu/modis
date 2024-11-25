#!/usr/bin/env python
# coding: utf-8 
# regression parameter optimization

# Author: farouk.lemmouchi@lisa.ipsl.fr
# This code along with the data are a demo to showcase the CHIMERE bias correction models capability
# developped and described in the paper

# "Machine learning-based improvement of aerosol optical depth from CHIMERE simulations using MODIS satellite observations"
# F. Lemmouchi ¹, J. Cuesta ¹, M. Lachatre ², J. Brajard ³, A. Coman ¹, M. Beekmann ¹, C. Derognat ²
# ¹ LISA, UMR 7583 CNRS, Université Paris-Est Créteil
# ² ARIA Technologies, Boulogne-Billancourt, France
# ³ Nansen Environmental and Remote Sensing Center (NERSC), Bergen, Norway

# THE SOFTWARE IS FOR R&D PURPOSES AND PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT.

#import tensorflow as tf; print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#import cuml

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,  minmax_scale
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA
from aeros5py.utils import count_days
import gc, random
from time import time 
from random import shuffle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import scipy, pickle
import yaml, os, sys

seed = 10 
fontsize=10
projection=ccrs.PlateCarree()

save_output=True
plot_dates=True
plot_average=True
# LOAD CONFIG
with open(sys.argv[1]) as f :
    config = yaml.load(f)
config['test_n'] = sys.argv[1].split('_')[1].split('.')[0]; config['test_n'] = 4
print(config['test_n'])

test_dates = config['test_dates']
train_dates = config['train_dates']

print('train_dates :', train_dates)
wv = config['wv']
which_modis = config['which_modis']
model_type=sys.argv[2]

columns = ["lon", "lat", "aod_550"] + config["var_2d_ch"] 
for v3d in config["var_3d_ch"] :
    for i in range(4) :
        columns.append(v3d + str(i))


if save_output :
#  input_layer, reference_layer = load_input_output2(date=train_dates[0], **config, common_na_mask=True, plot_corr=False,
#                                                  do_modis=True, do_chimere=True) # do not remove this line
#  The rest of the days
#  for date in train_dates[1:] :
#    print(date, end=', ')
#    #try :
#    inp, out = load_input_output2(date=date, **config, common_na_mask=True,
#                                                  do_modis=True, do_chimere=True) # do not remove this line
#
#    input_layer = np.append(input_layer, inp, axis=0)
#    reference_layer = np.append(reference_layer, out, axis=0)
#    #except :
  input_layer = pickle.load(open(f"input_layer_config_{config['test_n']}.pkl", 'rb'))
  reference_layer = pickle.load(open(f"reference_layer_config_{config['test_n']}.pkl", 'rb'))

  dataset = input_layer
  dataset.insert(dataset.shape[1], value=reference_layer.iloc[:,2], column="MODIS")

  dataset = dataset[dataset.lat>10]
  #### llllllllllllllllllllll Remove lon lat lllllllllllllllllll
  dataset = dataset.iloc[:,2:]
  dataset.reset_index(drop=True, inplace=True)
  dataset = dataset.sample(frac=1, random_state=3214) # only for linear regression
  halfsiz = int(dataset.index.size/2)

  x_train, y_train = dataset.iloc[:halfsiz,:-1].values, dataset.iloc[:halfsiz,-1].values.reshape(-1,1)
  x_test, y_test = dataset.iloc[halfsiz:,:-1].values, dataset.iloc[halfsiz:,-1].values.reshape(-1,1)

  input_layer = dataset.iloc[:,:-1].values; reference_layer = dataset.iloc[:,-1].values.reshape(-1,1)
  print("\nInput layer ", input_layer.shape)
  print("\nReference layer ", reference_layer.shape)
 
  print(model_type)
  if model_type == "lad":
    pass
  elif model_type == "linear":
    model = linear_model.LinearRegression(n_jobs=-1, positive=True)
  elif model_type == "random_forest" :
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(bootstrap=True, n_estimators=100, n_jobs=-1, random_state=seed, criterion="mse")
  elif model_type == "tpot" :
    from tpot import TPOTRegressor
    model = TPOTRegressor(verbosity=2,
                  generations=20,
                  population_size=20,
                  random_state=seed,
                  config_dict="TPOT cuML",
                  n_jobs=1,
                  cv=2,
                 )
    model.fit(x_test[0:20000,:], y_test[0:20000,:])
    model.export('tpot_code_on10k_3.py')
    sys.exit()
  elif model_type == "deepencoder" :
    from sklearn.model_selection import train_test_split
    x_train, x_test = train_test_split(input_layer, random_state=seed, shuffle=True)
    y_train, y_test = train_test_split(reference_layer, random_state=seed, shuffle=True)

    import tensorflow as tf; print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    def build_model(hp):
        model = tf.keras.Sequential()
 
        model.add(
            tf.keras.layers.BatchNormalization( 
                     input_shape=(x_train.shape[1],),
                     #activation='relu',
                     #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                     #bias_initializer="zeros",
            )
        )
    
    
        for l in range(hp.Int("nlayers", min_value=4, max_value=10, step=1)) :
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(f"units_{l}", min_value=10, max_value=50, step=5),
                    activation="relu",
                    #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    #bias_initializer="zeros",
                )
        )
        model.add(tf.keras.layers.Dense(1,
                    activation='relu',
                    #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    #bias_initializer="zeros",
                 )
        )
      
    
        model.compile(
    #         optimizer=tf.keras.optimizers.Adam(
    #             hp.Choice("learning_rate", values=[1e-2, 1e-3])
    #         ),
            optimizer=tf.keras.optimizers.Adam(), #learning_rate=1e-3),
            #loss="mean_squared_error",
            loss="mean_absolute_error",
            #metrics=["mean_absolute_error"],
        )
        return model
    from keras_tuner import RandomSearch
    from keras_tuner import BayesianOptimization

    class MyTuner(BayesianOptimization):
    #class MyTuner(RandomSearch):
        """
        https://github.com/keras-team/keras-tuner/issues/122
        """
        def run_trial(self, trial, *args, **kwargs):
            # You can add additional HyperParameters for preprocessing and custom training loops
            # via overriding `run_trial`
            #kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 5000, 50000, step=1000)
            #kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)
            super(MyTuner, self).run_trial(trial, *args, **kwargs)

    #tuner = BayesianOptimization (
    #tuner = RandomSearch(
    tuner = MyTuner(
        build_model,
        objective="val_loss",
        max_trials=50, # models
        executions_per_trial=1,
        overwrite=True,
        directory="models",
        project_name="tuner1",
        seed=seed
    )



    early_stop = tf.keras.callbacks.EarlyStopping(
                     monitor='val_loss',
                     min_delta=0.01, patience=3, verbose=0,
                     mode='min', baseline=300, restore_best_weights=True
                 )
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="models/log")
    
    start = time()
    tuner.search(x_train, y_train,
                 epochs=100,
                 batch_size=x_train.shape[0],
                 validation_data=(x_test, y_test),
                 verbose=3,
                 shuffle=False,
                 callbacks=[early_stop, tensorboard ],
                 #callbacks=[tensorboard ],
                )
    end = time()
    cost = np.round( (end - start)/60 , 2)
    print(f"NAC finished in : {cost} minutes" ) 

    model = tuner.get_best_models(1)[0]
    model.summary()

  start = time()
  if model_type not in ['deepencoder'] :
    model.fit( input_layer,reference_layer.flatten())
  end = time()
  cost = np.round( (end - start)/60 , 2)
  print(f"FT finished in : {cost} minutes" )
  

def save_prediction_date(date, model, pca, do_plot=None, save_output=None):
    print(date, end=', ')
    #input_layer_ver0, reference_layer_ver = load_input_output2(date=date, do_chimere=True, do_modis=True, common_na_mask=False, **config)
    input_layer_ver0, reference_layer_ver = load_input_output(date=date, do_chimere=True, do_modis=True, common_na_mask=False, **config)

    # Removing the equator region coz of low fire emissions issue 
    reference_layer_ver = reference_layer_ver[reference_layer_ver[:,1] > 10]
    input_layer_ver0 = input_layer_ver0[input_layer_ver0[:,1] > 10]
    chim_aod = np.copy(input_layer_ver0[:,2])

    #input_layer_ver0[:,2:] = scaler.transform(input_layer_ver0[:,2:])
 
    if input_layer_ver0.shape[0] < 200 : # low exploitable data points
        #continue
        return None, None, None, None
    #Predict on CHIMERE unfiltered
    input_layer_ver = input_layer_ver0 #input_layer_ver = minmax_scale(input_layer_ver0, feature_range=(-1, 1), axis=0)
    #input_layer = normalize(input_layer.T)
    prediction = model.predict(input_layer_ver0[:,2:])
    prediction = prediction.reshape(prediction.size,1)
 
    if save_output :
        out_pred_ds = xr.Dataset()
        out_pred_ds.coords['lon'] = (['south_north','west_east'], input_layer_ver0[:,0].reshape(63,-1)) # reshape cz this is subdomain 10°N threshold
        out_pred_ds.coords['lat'] = (['south_north','west_east'], input_layer_ver0[:,1].reshape(63,-1))
        out_pred_ds[f'optdaero_{wv}'] = (['south_north','west_east'], chim_aod.reshape(63,-1))
        out_pred_ds[f'optdaero_{wv}_corr'] = (['south_north','west_east'], prediction.reshape(63,-1))

        out_pred_ds.to_netcdf( f'{output_path}/out.{date}00_01.nc', mode='w')
        out_pred_ds.close()
    return 

# #### plot_testing_dates
def save_prediction_dates (test_dates, model, pca, do_plot=False, save_output=False) :
    log_reference = pd.DataFrame()
    log_prediction = pd.DataFrame()
    inference_cost = np.empty(0)

    for date in test_dates :
        start = time()
        #metrics, metrics0 =  predict_date(date, model, pca, do_plot=do_plot, save_output=save_output)
        save_prediction_date(date, model, pca, do_plot=do_plot, save_output=True)
        end = time()
        #inference_cost = np.append(inference_cost, end-start)
        #log_reference   = log_reference.append(metrics0, ignore_index=True)
        #log_prediction = log_prediction.append(metrics, ignore_index=True)
    return log_reference, log_prediction, inference_cost

if save_output :
   ## Run plot 
  figpath=f"./figures/{config['region']}/{model_type}/{str(config['test_n'])}/"
  output_path = f"/media/disk1/output/{config['region']}/{model_type}/{config['test_n']}/"
  if not os.path.exists(figpath) : os.mkdir(figpath)
  if not os.path.exists(output_path) : os.mkdir(output_path)

  ##log_reference, log_prediction, inference_cost = plot_testing_dates(test_dates, model, None, do_plot=False, save_output=True)
  log_reference, log_prediction, inference_cost = save_prediction_dates(test_dates, model, None, do_plot=False, save_output=True)


  # MODEL INFO
  f = open(f"{figpath}/config_{config['test_n']}.yaml", 'w+')
  yaml.dump(config, f, allow_unicode=True)
  yaml.dump(
      {
          "model_type": model_type,
          #"test_dates" : test_dates,
          #"train_dates" : train_dates,
          #"Daily_average_CHIMERE.raw_vs_MODIS":np.float(np.sum(log_reference)/len(test_dates)),
          #"Daily_average_testing_score":np.float(np.sum(log_prediction)/len(test_dates)),
          #"train_cost_minutes":np.float(cost),
          #"Daily_average_prediction_cost_minutes":float(np.sum(inference_cost)/60/len(test_dates)),
          "figpath":figpath,
          "output_path":output_path,
          
      }, f, default_flow_style=False)
  if model_type == "knn":
      yaml.dump(config, f, allow_unicode=True)
      yaml.dump({
          "k":k,
          "npca":nclusters}, f, default_flow_style=False)





if plot_average :
  #### AVERAGE #####
  with open(f"/home/farouk/post_proc/AI/ai-pp/figures/{config['region']}/{model_type}/{config['test_n']}/config_{config['test_n']}.yaml") as f :
      config = yaml.load(f)
  dates = config['test_dates']

  # ### orig vs corr

  dates = config['test_dates']
  a = []
  b = []
  from aeros5py.post_proc import convert_aod_wv
  for date in dates :
      print(date, end=",")
      chim_ds = xr.open_dataset(f"{config['output_path']}/out.{date}00_01.nc")

      o = chim_ds.lat>10
      a.append(chim_ds['optdaero_550.0'].where(o).data)
      b.append(chim_ds['optdaero_550.0_corr'].where(o).data)


  # In[4]:


  chim_orig = np.asarray(np.squeeze(a))
  chim_corr = np.asarray(b)
  diff = chim_corr - chim_orig


  # In[5]:


  fig = plt.figure(figsize=(20,10))
  ax = fig.add_subplot(projection=ccrs.PlateCarree())
  plt.pcolormesh(chim_ds.lon, chim_ds.lat, np.nanmean(diff, axis=0), cmap="bwr", vmin=-1.2, vmax=1.2)# cmap='YlOrBr') #, vmin=0, vmax=1)
  ax.coastlines()
  plt.title(f"AOD average difference CHIMERE_corrected-raw on testing_dates {config['test_n']}", fontsize=fontsize)
  plt.ylim([10, chim_ds.lat.max()])
  plt.colorbar()
  ax, txt = set_metrics_str(ax, chim_orig.flatten(), chim_corr.flatten(),  fontsize=fontsize ) # lat 10 == idx 37


  plt.text(-14,5,f"Applied on impair months", fontsize=fontsize)
  plt.text(7,5,f" Model: {config['model_type']}", fontsize=fontsize)
  plt.text(-14,3,f" Vars: AOD_550, {', '.join(config['var_3d_ch'] + config['var_2d_ch'] )}", fontsize=fontsize)

  plt.savefig(f"{config['figpath']}/{config['model_type']}_average_difference_{config['test_n']}.jpg")


  # ### modis vs corr

  # In[6]:


  a = []
  b = []
  for date in dates :
      print(date, end=",")
      chim_ds = xr.open_dataset(f"{config['output_path']}/out.{date}00_01.nc")
      modis_ds = load_datasets(date=date, do_chimere=False, do_modis=True, **config)
      o = modis_ds.lat>10
      a.append(chim_ds['optdaero_550.0_corr'].data)    
      b.append(modis_ds.AOD_550_Dark_Target_Deep_Blue_Combined.where(o).data[37:,:])


  # In[9]:


  chim_corr = np.asarray(np.squeeze(a))
  modis_aod = np.asarray(b)
  bias = modis_aod - chim_corr


  # In[10]:

  fig = plt.figure(figsize=(20,10))
  ax = fig.add_subplot(projection=ccrs.PlateCarree())
  plt.pcolormesh(chim_ds.lon, chim_ds.lat, np.nanmean(bias, axis=0), cmap="bwr", vmin=-1.2, vmax=1.2)# cmap='YlOrBr') #, vmin=0, vmax=1)
  ax.coastlines()
  plt.title(f"AOD average bias MODIS-corrected on testing_dates, test {config['test_n']}", fontsize=fontsize)
  plt.ylim([10,chim_ds.lat.max()])
  plt.colorbar()
  ax, txt = set_metrics_str(ax, chim_corr.flatten(), modis_aod.flatten(), fontsize=fontsize ) # lat 10 == idx 37
  #ax, txt = set_metrics_str(ax,optdaero.flatten(), modis_aod.flatten(), fontsize=fontsize )


  plt.text(-14,5,f"Applied on impair months", fontsize=fontsize)
  plt.text(7,5,f" Model: {config['model_type']}", fontsize=fontsize)
  plt.text(-14,3,f" Vars: {', '.join(config['var_3d_ch'] + config['var_2d_ch'] )}", fontsize=fontsize)

  plt.savefig(f"{config['figpath']}/{config['model_type']}_average_bias_{config['test_n']}.jpg")



if plot_dates :
  # plot performances panel panel figures 
  import matplotlib as mpl
  mpl.style.use('ggplot')
  _ = plot_testing_dates(config['test_dates'], config=config)
