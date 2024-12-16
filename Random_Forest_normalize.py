# 正規化したうえでRandom Forestを行う
# environment

import tensorflow as tf; print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#import cuml
import matplotlib.pyplot as plt

plt.style.use("ggplot")

import warnings;  warnings.filterwarnings('ignore')
seed = 10
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
#from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from sklearn.preprocessing import normalize,  minmax_scale
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA, SparsePCA
from lib.aeros5py.utils import count_days
import gc, random, datetime    
from time import time 
#from random import shuffle
#from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import scipy
import yaml, os

test_n  = 4 # for Lemmouchi et al., 2023 Remote Sensing

# parameter
import cartopy.crs as ccrs
import pickle
test_n = 4

variables = dict()
figsize=(14,7)
seed = 10
wv = "550.0"
projection=ccrs.PlateCarree()
variables["var_2d_modis"] = ["lon", "lat", "AOD_550_Dark_Target_Deep_Blue_Combined"]

#variables["var_2d_ch"] = [ "lon", "lat", "albedo", "copc", "lspc", "pblh", "sshf", "swrd", "tem2", "topc", "u10m", "usta", "v10m", "ws10m", "wsta", "slhf", "soim", "sreh", "aerr", "atte"] 
#variables["var_2d_ch"] = ["hght", "albedo", "usta", "wsta"] # deepencoder & random forest GOOOOOOOD
#variables["var_2d_ch"] = ["albedo","swrd","topc","soim","sreh", "wsta","usta","hght","slhf","sshf"] #"lspc", "copc", #,"atte"] # full
#variables["var_2d_ch"] = ["albedo","swrd","soim","sreh", "hght"] #"lspc", "copc", #,"atte"] # full 69
#variables["var_2d_ch"] = ["lspc", "copc", #,"atte"] # full

# Note: aod_550 should take always the first rank if used.
#variables["var_2d_ch"] = ["albedo","swrd","sreh", "slhf","sshf"] #"hght","lspc", "copc", #,"atte"] # full
variables["var_2d_ch"] = ["albedo","swrd","soim","sreh", "hght","slhf","sshf"] #"lspc", "copc", #,"atte"] # full

# bad variables : albedo
#variables["var_2d_ch"] = ["lon", "lat", "optdaero_550.0", "hght", "albedo","tem2","u10m","v10m"] # deepencoder & random forest

#variables["var_2d_ch"] = ['lon', 'lat', 'optdaero_550.0', 'pblh', 'usta', 'wsta', 'tem2', 'v10m', 'u10m', 'ws10m']
#variables["var_3d_ch"] = ['PM10', 'PM25', 'temp', 'winm', 'winw', 'pres']
#variables["var_3d_ch"] = ['PM10', 'temp', 'pres'] #'winm', 'winw', 'pres']

#variables["var_3d_ch"] = [ "pres", "rain", "relh", "snow", "sphu", "temp", "thlay", "winm", "winz", "winw",
#                      "PM25bio", "PM25ant", "PM25", "O3", "NO2", "NO", "PAN", "HNO3", "H2O2", "H2SO4",
#                      "HONO_CHEM", "HONO_AER", "HONO_SURF", "HONO_ANT", "HONO_FIRE", "SO2", "CO", "OH", "CH4", "C2H6", "NC4H10",
#                      "C2H4", "C3H6", "OXYL", "C5H8", "HCHO", "CH3CHO", "GLYOX", "MGLYOX", "CH3COE", "NH3", "APINEN", "BPINEN", "LIMONE", "TERPEN", "OCIMEN",
#                      "HUMULE", "TOL", "TMB", "AnA1D", "AnBmP", "BiA1D", "BiBmP", "ISOPA1", "PPMAQ", "H2SO4AQ", "HNO3AQ", "NH3AQ", "AnA1DAQ", "AnBmPAQ",
#                      "BiA1DAQ", "BiBmPAQ", "ISOPA1AQ", "HCL", "HCLAQ", "NAAQ", "DUSTAQ", "OCARAQ", "BCARAQ", "DMS", "DMSO",
#                      "MSA", "TOTPAN", "NOX", "OX", "HOX", "NOY", "ROOH", "HCNM", "pBCAR", "pDUST", "pNA", "pOCAR", "pPPM", "pAnA1D", "pAnBmP", "pBiA1D",
#                      "pBiBmP", "pISOPA1", "pH2SO4", "pHCL", "pHNO3", "pNH3", "pWATER", "PM25", "PM10", "PM25", "PM10bio", "PM10ant", "chem_regime", "airm",
#                      "CloudpH", "AeropH", "kzzz", "cliq", "cice", "dpeu", "dpdu", "dpdd", "flxu", "flxd", "jO3", "jNO2"  ]
#variables["var_3d_ch"] = ["pBCAR","pOCAR", "pDUST", "PM10", "PM25", "O3", "NO2", "temp", "relh", "pres", "winm", "winw", "winz"] #deepencoder & randomforest GOOOOOOOD
#variables["var_3d_ch"] = ["PM10","PM25","pDUST","pBCAR","CO","NO2","O3","pres","temp","relh","winm","winz","winw"] # presentation Reunion 1

#variables["var_3d_ch"] = ["winz","winm","winw","pres","airm", "relh","temp","sphu","O3","SO2","CO","OH","CH4","NH3","TOL","NOX","HOX","NOY","ROOH","HCNM","pDUST","pOCAR","pSALT","pH2SO4","pHNO3","pNH3","pWATER"] #full

#variables["var_3d_ch"] = ["temp", "pHNO3", "sphu", "pH2SO4", "OH"] #minimal

variables["var_3d_ch"] =  ["pres","relh","ROOH","HCNM","pSALT","pH2SO4","pHNO3","pNH3","pWATER","pDUST","pOCAR","sphu", "O3","SO2","CO","OH","NH3","TOL","NOX","NOY",] # Lemmouchi et al. 2023 ?

##########variables["var_3d_ch"] = ["PM10", "PM25", "pDUST","pOCAR","pWATER","pSALT","pres","relh","ROOH","HCNM","pH2SO4","pHNO3","pNH3","sphu","O3","SO2","CO","OH","NH3","TOL","NOX","NOY"]
#variables["var_3d_ch"] = ["PM10", "PM25", "pDUST","pOCAR","pWATER", "winz", "winm", "winw"]

# columns = [ "lon", "lat"] + variables["var_2d_ch"] 
# for v3d in variables["var_3d_ch"] :
#     for i in range(4) :
#         columns.append(v3d + str(i))

#variables["var_3d_ch"] = ["O3","pDUST","pOCAR","pres"] ,
#variables["var_3d_ch"] = ["PM10","relh", "pres"] #deepencoder & randomforest  f"",
#variables["var_3d_ch"] = ["PM10", "PM25", "pBCAR", "pOCAR", "O3", "NO2", "temp", "relh", "pres", "winm", "winw", "winz"] #deepencoder & randomforest 
#variables["var_3d_ch"] = [] #"PM10", "PM25"] #"PM10"] #pBCAR","pOCAR", "pDUST", "PM10", "PM25", "O3", "NO2", "temp", "relh", "pres", "winm", "winw", "winz"] #, ]

#variables["var_3d_ch"] = []

which_inst = "AQUA"
config = {
    "wv" : "550.0", #200 300 400 600 999
    "aod_threshold" : None, # (0,0.6),  #(-10., 70),
    "res" : 0.45,
    "hour" : '13',
    "which_inst" : which_inst,
    "region" : "AFRICA",
    "var_3d_ch" : variables["var_3d_ch"],
    "var_2d_ch" : variables["var_2d_ch"],
    "var_2d_modis" : variables["var_2d_modis"],
    "version" : "v23",
    "test_n": test_n,
}

#print(columns)

# Training dates selection

# 1/3 first days
train_dates = [] #count_days('20210510','2020531','%Y%m%d')
for month in range(1,13,1) :
    train_dates = train_dates + count_days(f'2021{str.zfill(str(month),2)}01',f'2021{str.zfill(str(month),2)}20','%Y%m%d')

tmp_dates = count_days('20210101','20211231','%Y%m%d')

for d in train_dates + ["20210923", "20210925", "20210924"] : #, "20211014"] # no modis on 23-25 Sep 2021
    tmp_dates.remove(d)
valid_dates = tmp_dates

all_dates = count_days('20210101','20211231','%Y%m%d')
for d in ["20210923", "20210925", "20210924"] : #, "20211014"] # no modis on 23-25 Sep 2021
    all_dates.remove(d)
    
print(len(train_dates), len(valid_dates), len(all_dates) )
config['train_dates'] = train_dates
config['valid_dates'] = valid_dates

output_path0 = f'./models/{config["region"]}/'
if not os.path.exists(output_path0) : os.mkdir(output_path0)

# SAVE CONFIG
f = open(f'{output_path0}/config_{config["test_n"]}.yaml', 'w+')
yaml.dump(config, f, allow_unicode=True)

# # LOAD CONFIG
# with open(f"config_{test_n}.yaml") as f :
#     config = yaml.load(f)

## Input/Output setup
# Extracting
# Train data(1か月のうち1日 ~ 20日)
input_layer_train, reference_layer_train = load_input_output2(date=train_dates[0], common_na_mask=True, plot_corr=False,
                                                do_modis=True, do_chimere=True, **config) # do not remove this line
input_layer_train ['date'] = train_dates[0]

# The train days
for date in train_dates[1:] :
  print(date, end=', ')
  #try :
  inp, out = load_input_output2(date=date, **config, common_na_mask=True,
                                                do_modis=True, do_chimere=True) # do not remove this line
  inp['date'] = date
  input_layer_train = input_layer_train.append(inp)
  reference_layer_train = reference_layer_train.append(out)
  #except :
print("\nInput layer train", input_layer_train.shape)
print("\nReference layer train", reference_layer_train.shape)

# Create dataset with both CHIMERE and MODIS common data for train dates

dataset = input_layer_train
dataset.insert(input_layer_train.shape[1], value=reference_layer_train.iloc[:,2].values, column="MODIS")

print("\nCoincident CHIMERE/MODIS data")
print("\nInput layer train", input_layer_train.shape)
print("\nReference layer train", reference_layer_train.shape)
print("\n")

# The rest of the days (CHIMERE data without coincidence with MODIS)

for date in train_dates[:] :
  print(date, end=', ')
  #try :
  inp, out = load_input_output2(date=date, **config, common_na_mask=False,
                                                do_modis=True, do_chimere=True) # do not remove this line
  inp['date'] = date
  input_layer_train = input_layer_train.append(inp)
  reference_layer = reference_layer_train.append(out)
  #except :

print("\n All training dates & Including CHIMERE only pixels => dataset ")
print("\nInput layer train", input_layer_train.shape)
print("\nReference layer train", reference_layer_train.shape)
print("\n")

# Train dataで使われた日以外はValidation data として扱う
# The validation days
input_layer_valid, reference_layer_valid = load_input_output2(date=valid_dates[0], common_na_mask=True, plot_corr=False,
                                                do_modis=True, do_chimere=True, **config) # do not remove this line
input_layer_valid['date'] = valid_dates[0]

for date in valid_dates[1:] :
  print(date, end=', ')
  #try :
  inp, out = load_input_output2(date=date, **config, common_na_mask=True,
                                                do_modis=True, do_chimere=True) # do not remove this line
  inp['date'] = date
  input_layer_valid = input_layer_valid.append(inp)
  reference_layer_valid = reference_layer_valid.append(out)

dataset_valid = input_layer_valid
dataset_valid.insert(input_layer_valid.shape[1], value=reference_layer_valid.iloc[:,2].values, column="MODIS")

#date1 = '20210930'
#a = input_layer_valid[input_layer_valid['date'] == date1]
#print(a)
#print(a.shape)

for date in valid_dates[:] :
  print(date, end=', ')
  #try :
  inp, out = load_input_output2(date=date, **config, common_na_mask=False,
                                                do_modis=True, do_chimere=True) # do not remove this line
  inp['date'] = date
  input_layer_valid = input_layer_valid.append(inp)
  reference_layer_valid = reference_layer_valid.append(out)

print("\nIncluding CHIMERE only pixels & Validation dates => dataset_valid ")
print("\nInput layer valid ", input_layer_valid.shape)
print("\nReference layer valid ", reference_layer_valid.shape)
print("\n")

dataset_valid = input_layer_valid

# Saving input_layer and reference_layer
import pickle

output_path0 = f"./models/{config['region']}/"
if not os.path.exists(output_path0) : os.mkdir(output_path0)

pickle.dump(input_layer_train, open(f"./models/{config['region']}/input_layer_train_{config['test_n']}.pkl", 'wb'))
pickle.dump(reference_layer_train, open(f"./models/{config['region']}/reference_layer_train_{config['test_n']}.pkl", 'wb'))

pickle.dump(input_layer_valid, open(f"./models/{config['region']}/input_layer_valid_{config['test_n']}.pkl", 'wb'))
pickle.dump(reference_layer_valid, open(f"./models/{config['region']}/reference_layer_valid_{config['test_n']}.pkl", 'wb'))

#input_layer_train = pickle.load(open(f"./models/{config['region']}/input_layer_train_{config['test_n']}.pkl", 'rb'))
#reference_layer_train = pickle.load(open(f"./models/{config['region']}/reference_layer_train_{config['test_n']}.pkl", 'rb'))

# Removing the equator region coz of low fire emissions issue
dataset = dataset[dataset.lat>10]
#### llllllllllllllllllllll Remove lon lat lllllllllllllllllll
dataset = dataset.iloc[:,2:]

# #Remove MODIS pixels that are AOD limited to 3.5
# idx0 = (dataset.MODIS < 3.45) | (dataset.MODIS > 3.51)
# plt.scatter(dataset.aod_550[idx0], dataset.MODIS[idx0], 1, c='k')
# plt.show()
# plt.scatter(input_layer[:,0], reference_layer, 2)
#dataset = dataset[idx0]

# Train/test data splitting
dataset = dataset.sample(frac=1, random_state=3214)
halfsiz = int(dataset.index.size/2)

#x_train, y_train = dataset.iloc[:halfsiz,:-1].values, dataset.iloc[:halfsiz,-1].values.reshape(-1,1)
#x_test, y_test = dataset.iloc[halfsiz:,:-1].values, dataset.iloc[halfsiz:,-1].values.reshape(-1,1)
x_train, y_train = dataset.iloc[:halfsiz,:-2].values, dataset.iloc[:halfsiz,-1].values.reshape(-1,1) # -2 for withdrawing the date & MODIS by Juan
x_test, y_test = dataset.iloc[halfsiz:,:-2].values, dataset.iloc[halfsiz:,-1].values.reshape(-1,1) # -2 for withdrawing the date & MODIS by Juan

#from sklearn.model_selection import train_test_split
#x_train, x_test = train_test_split(input_layer, random_state=seed, shuffle=True)
#y_train, y_test = train_test_split(reference_layer, random_state=seed, shuffle=True)

input_layer = dataset.iloc[:,:-1].values; reference_layer = dataset.iloc[:,-1].values.reshape(-1,1)

# Saving DTrain & DTest by Juan
DTrain = dataset.iloc[:halfsiz,:]
DTest = dataset.iloc[halfsiz:,:]

pickle.dump(DTrain, open(f"./models/{config['region']}/DTrain_{config['test_n']}.pkl", 'wb'))
pickle.dump(DTest, open(f"./models/{config['region']}/DTest_{config['test_n']}.pkl", 'wb'))

# データを正規化する(まずはfeature vectorだけ)
def normalize_data(X_train, X_test):

    # normalize X 
    dataScalerX = sklearn.preprocessing.RobustScaler().fit(X_train) # 学習データで各説明変数の中央値とIQRを求める
    X_train = dataScalerX.transform(X_train)
    X_test = dataScalerX.transform(X_test)

    return X_train, X_test

x_train, x_test = normalize_data(x_train, x_test)

## Random forest
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Thanks to this post https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
model_type = "random_forest"
# model selection
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 200, num = 50)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [] #[int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2] #, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf =  [1] #, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    'n_estimators': n_estimators,
    'max_features': ['sqrt', 'log2'],
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap
}

# Create a based model
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=1,
                               random_state=seed,
                               n_jobs=2) # n_jobs=-1 for all available processors

#rf_random.fit(x_train[:100,:], y_train[:100].flatten()) # first 100 elements - small test
#rf_random.fit(input_layer, reference_layer.flatten()) # full run

rf_random.fit(x_train[:10000,:], y_train[:10000].flatten()) # only 10 000 pixels
print(rf_random.best_estimator_)

#del grid_search, rf
# Grid search CPU
model_type = "random_forest"

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [False, True],
    'random_state':[seed],
    'max_features': ['sqrt'], #, 'log2'],
    'min_samples_leaf':[2], # , 4, 6],
    'min_samples_split':[2], #, 4, 6],
    'n_estimators': [100] #list(np.arange(20) ) #, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                  cv=2, n_jobs=2, verbose=3) # n_jobs=-1 for all available processors
start= time()
#grid_search.fit(input_layer, reference_layer.flatten()) # full dataset / too long
grid_search.fit(x_train[:10000,:] , y_train[:10000].flatten()) # only 10 000 pixels

end= time()
cost = np.round( (end - start)/60 , 2)
print(f"FT finished in : {cost} minutes")

#rf = cuml.RandomForestRegressor()

## Grid search GPU
#model_type = "random_forest"

#from cuml.model_selection import GridSearchCV
## Create the parameter grid based on the results of random search
#param_grid = dict(                 random_state=[seed], 
#                                   #n_streams=[],
#                                   #split_criterion=split_criterion, 
#
#                                   max_features=['auto'], n_bins=[64, 128],
#                                   min_samples_leaf=[2, 4, 6],
#                                   min_samples_split=[2, 4, 6],
#                                   n_estimators=[100])
## Create a based model
#rf = cuml.RandomForestRegressor()
## Instantiate the grid search model
#grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_squared_error',
#                          cv=2, verbose=2)
#start= time()
##grid_search.fit(np.array(input_layer, dtype=np.float32), np.array(reference_layer,dtype=np.float32))
#grid_search.fit(input_layer, reference_layer)
#end = time()
#
#cost = np.round( (end - start)/60 , 2)
#print(f"FT finished in : {cost} minutes")

model0 = grid_search.best_estimator_
model = grid_search.best_estimator_

#model = model0.convert_to_fil_model()
#{'max_features': 20, 'min_samples_leaf': 4, 'min_samples_split': 2, 'n_bins': 128, 'n_estimators': 100, 'random_state': 10}
print(grid_search.best_params_)

model_type = "random_forest"
# Create a based model
model = RandomForestRegressor(bootstrap=True, n_estimators=100, n_jobs=2, 
                              random_state=seed, min_samples_leaf=4, 
                              max_features=20)
model.fit(x_train, y_train.flatten())

prediction = model.predict(x_test)

#_, metrics = set_metrics_str(ax, y_test.flatten(), prediction.flatten())

plt.plot(model.feature_importances_)

ft = np.flipud(np.argsort(model.feature_importances_))
np.array(dataset.columns[:-1])[ft]

model.feature_importances_.mean()

idx = np.where(model.feature_importances_> 0.02)

fig, ax = plt.subplots(figsize=(10,5))
plt.bar(dataset.columns[:-1][idx], model.feature_importances_[idx])

plt.yscale('log')
_ = plt.xticks(rotation=45)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig("rf_features_importance.jpg", dpi=100, pad_inches=12)

idx = np.where(model.feature_importances_> 0.02)

fig, ax = plt.subplots(figsize=(10,5))
plt.bar(dataset.columns[:-1][idx], model.feature_importances_[idx])

plt.yscale('log')
_ = plt.xticks(rotation=45)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.savefig("rf_features_importance.jpg", dpi=100, pad_inches=12)

prediction = model.predict(x_test)
print(x_test.shape)
MAE = mean_absolute_error(y_test.flatten(), prediction)
print(f'RF validation MAE = {MAE}')

model_type = "RF"
# global figpath, which_inst, output_path
figpath=f"./figures/{config['region']}/{model_type}/{str(test_n)}/"
output_path = f"./models/{config['region']}/{model_type}/{str(test_n)}/"
if not os.path.exists(figpath) : os.mkdir(figpath)
if not os.path.exists(output_path) : os.mkdir(output_path)

pickle.dump(model, open(f'{output_path}/{model_type}.pkl', 'wb'))

#ax = plt.subplot()
#model_type = "random_forest"

#metrics_list = []

#model = cuml.RandomForestRegressor(accuracy_metric="mse",
#                                   random_state=seed,
#                                   n_streams=1,
#                                   max_features=1.0, n_bins=256,
#                                   min_samples_leaf=1,
#                                   min_samples_split=2,
#                                   #split_criterion=split_criterion,
#                                   n_estimators=200) #n_bins=256)

#start = time()
##model.fit(x_train,y_train.flatten())
#model.fit(input_layer,reference_layer.flatten())

#end = time()
#cost = np.round( (end - start)/60 , 2)
#print(f"FT finished in : {cost} minutes" ) 

## Get the mean absolute error on the validation data
#prediction = model.predict(x_test)

#_, metrics = set_metrics_str(ax, y_test.flatten(), prediction.flatten())
#metrics_list.append(metrics)
#print(pd.DataFrame.from_dict(metrics_list))
#plt.close()